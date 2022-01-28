import os, sys

import numpy as np
import math, random
import copy as cp

import ModelHelper.globals as g


MAX_MUTATE_ATTEMPT=10000

"""
# chromosome format should be in following
# {
    layers:  [[layer_type, p1, p2, p3], ...(n times)]
    opt:    optimizer
    1d_start: layer# of first 1d layer
# }
"""
"""
# For each layer_type
###################/###################/###################
## Conv2D          /## AveragePooling2D/## MaxPooling2D
# p1 = activation  /# p1 = dc          /# p1 = dc
# p2 = filters     /# p2 = pool_size   /# p2 = pool_size
# p3 = kernel_size /# p3 =             /# p3 =
###################/###################/###################
## Dense           /## Dropout
# p1 = activation  /# p1 = dc
# p2 = units       /# p2 =
# p3 =             /# p3 = dropout range
###################/###################
"""
"""
In all honesty, i am too lazy to optimize the following..... so lets assume they are efficient enought
"""
# shorter random fuc call
def _gen_CONV2D_KERNEL():
    return random.randrange(g._L_CONV2D_KERNEL_RANGE[0], g._L_CONV2D_KERNEL_RANGE[1]+1)
def _gen_POOLING_SIZE():
    return random.randrange(g._L_POOLING_SIZE_RANGE[0], g._L_POOLING_SIZE_RANGE[1]+1)
def _gen_CONV2D_FILTER():
    return random.randrange(g._L_CONV2D_FILTER_RANGE[0], g._L_CONV2D_FILTER_RANGE[1]+1)
def _gen_DENSE_UNIT():
    return random.randrange(g._L_DENSE_UNIT_RANGE[0], g._L_DENSE_UNIT_RANGE[1]+1)
def _gen_DROUPOUT_RANGE():
    return random.uniform(*g._L_DROUPOUT_RANGE)


def generate_chromosome(debug=False):
    while True:
        chromosome={}
        n_layer = random.randrange(g._CHROMO_LEN_RANG_min, g._CHROMO_LEN_RANG_max+1)
        _1d_start = random.randrange(g._CHROMO_LEN_RANG_min, g._CHROMO_LEN_RANG_max+1)
        chromosome['1d_start']=_1d_start
        chromosome['opt'] = random.choice(g._CHROMO_OPTMIZER_TYPE)

        layers=[]
        #generate 2d
        for ind in range(_1d_start):
            type = random.choice(list(g._CHROMO_2D_LAYER_TYPE))
            activation = random.choice(list(g._CHROMO_ACTIVATE_FNC_TYPE))
            p3 = _gen_CONV2D_KERNEL()
            if type in [1, 2]:
                p2 = _gen_POOLING_SIZE()
            else:
                p2 = _gen_CONV2D_FILTER()
            layers.append([type, activation, p2, p3])
        #generate 1d
        for ind in range(n_layer-_1d_start):
            type = random.choice(list(g._CHROMO_1D_LAYER_TYPE))
            activation = random.choice(list(g._CHROMO_ACTIVATE_FNC_TYPE))
            p2 = _gen_DENSE_UNIT()
            p3 = _gen_DROUPOUT_RANGE()
            layers.append([type, activation, p2, p3])

        chromosome['layers'] = layers

        if check_chromosome(chromosome, debug=debug):
            break
        assert iter<MAX_MUTATE_ATTEMPT, "Max mutate attempt reached"
    return chromosome


def mutate_chromosome(parent, debug=False):
    ## should mutate optimizer
    if random.random()<g._MUTATE_OPTMIZER_PROB:
        child = cp.deepcopy(parent)
        child['opt'] = random.choice(g._CHROMO_OPTMIZER_TYPE)
        return child
    #get gene location to change while enforcing the gene length parameter
    n_layer = len(parent['layers'])
    while True:
        if random.random()<g._LAYER_LEN_CHANGE_PROB:
            gene_loc = random.choice([-1, n_layer])
        else:
            gene_loc = random.randrange(0, n_layer)
        if n_layer == g._CHROMO_LEN_RANG_max and gene_loc == n_layer:
            continue
        if n_layer == g._CHROMO_LEN_RANG_min and gene_loc == -1:
            continue
        break
        #loop until valid gene_loc is produced
    iter=0
    while True:
        iter+=1
        child = cp.deepcopy(parent)
        ## apply mutation to gene
        if gene_loc == -1: # first layer will be removed
            new_layer = parent['layers'][1:]
            new_1d_start = parent['1d_start']-1
            if new_1d_start<0:
                new_1d_start=0
            child.update({
                'layers':new_layer,
                '1d_start':new_1d_start,
            })
        elif gene_loc == n_layer: # new layer will be appended
            type = random.choice(list(g._CHROMO_1D_LAYER_TYPE))
            activation = random.choice(list(g._CHROMO_ACTIVATE_FNC_TYPE))
            p2 = _gen_DENSE_UNIT()
            p3 = _gen_DROUPOUT_RANGE()
            child['layers'].append([type, activation, p2, p3])
        else:
            #mutate the given layer
            # if random.random()<g._MUTATE_LAYER_TYPE_PROB:
            old_type = parent['layers'][gene_loc][0]
            old_activation = parent['layers'][gene_loc][1]
            new_type = random.choice(list(g._CHROMO_LAYER_TYPE))
            #will mutate 1d to 2d layer
            if gene_loc == parent['1d_start'] and new_type in g._CHROMO_2D_LAYER_TYPE:
                new_1d_start = parent['1d_start']+1
                p3 = _gen_CONV2D_KERNEL()
                if new_type in [1, 2]:
                    p2 = _gen_POOLING_SIZE()
                else:
                    p2 = _gen_CONV2D_FILTER()
                child['layers'][gene_loc] = [new_type, old_activation, p2, p3]
                child['1d_start'] = new_1d_start
            #will mutate 2d to 1d layer
            elif gene_loc == parent['1d_start']-1 and new_type in g._CHROMO_1D_LAYER_TYPE:
                new_1d_start = parent['1d_start']-1
                p2 = _gen_DENSE_UNIT()
                p3 = _gen_DROUPOUT_RANGE()
                child['layers'][gene_loc] = [new_type, old_activation, p2, p3]
                child['1d_start'] = new_1d_start
            #just mutate the parameter
            #2d
            elif (old_type in g._CHROMO_2D_LAYER_TYPE) and (new_type in g._CHROMO_2D_LAYER_TYPE):
                activation = random.choice(list(g._CHROMO_ACTIVATE_FNC_TYPE))
                p3 = _gen_CONV2D_KERNEL()
                if new_type in [1, 2]:
                    p2 = _gen_POOLING_SIZE()
                else:
                    p2 = _gen_CONV2D_FILTER()
                child['layers'][gene_loc] = [new_type, activation, p2, p3]
            #1d
            elif (old_type in g._CHROMO_1D_LAYER_TYPE) and (new_type in g._CHROMO_1D_LAYER_TYPE):
                activation = random.choice(list(g._CHROMO_ACTIVATE_FNC_TYPE))
                p2 = _gen_DENSE_UNIT()
                p3 = _gen_DROUPOUT_RANGE()
                child['layers'][gene_loc] = [new_type, activation, p2, p3]


        if check_chromosome(child, debug=debug):
            break

        # if iter > MAX_MUTATE_ATTEMPT-1:
        #     print(iter, "trying at ", gene_loc, n_layer, parent)
            # return parent
        assert iter<MAX_MUTATE_ATTEMPT, "Max mutate attempt reached: %d"%iter

    return child



def check_chromosome(chromosome, debug=False):
    #check chromosome length
    if  len(chromosome['layers']) > g._CHROMO_LEN_RANG_max or \
        len(chromosome['layers']) < g._CHROMO_LEN_RANG_min:
        if debug:
            print("chromosome out of length")
        return False

    #check optimizer type
    if not chromosome['opt'] in g._CHROMO_OPTMIZER_TYPE:
        if debug:
            print("optimizer type error")
        return False

    d1_start=chromosome['1d_start']
    #check d1_start
    if len(chromosome['layers'])+1 < d1_start:
        if debug:
            print("d1_start outside of range")
        return False
    #check 2d layer should be before 1d layer
    for ind, layer in enumerate(chromosome['layers'][:d1_start]):
        if layer[0] in g._CHROMO_1D_LAYER_TYPE:
            if debug:
                print("1d layer before d1_start:", ind)
            return False
    for ind, layer in enumerate(chromosome['layers'][d1_start:]):
        if layer[0] in g._CHROMO_2D_LAYER_TYPE:
            if debug:
                print("2d layer after d1_start:", ind)
            return False

    #check parameter of each layer
    for ind, layer in enumerate(chromosome['layers']):
        p1, p2, p3 = layer[1:]
        # assume p1 is always right
        # Conv2D
        if layer[0]==0:
            if not p2 in list(range(g._L_CONV2D_FILTER_RANGE[0], g._L_CONV2D_FILTER_RANGE[1]+1)):
                if debug:
                    print("Conv2D FILTER out if range:", p2)
                return False
            if not p3 in list(range(g._L_CONV2D_KERNEL_RANGE[0], g._L_CONV2D_KERNEL_RANGE[1]+1)):
                if debug:
                    print("Conv2D k_size out if range:", p3)
                return False
        # AveragePooling2D or MaxPooling2D
        elif layer[0] in [1, 2]:
            if not p2 in list(range(g._L_POOLING_SIZE_RANGE[0], g._L_POOLING_SIZE_RANGE[1]+1)):
                if debug:
                    print("Pooling k_size out if range:", p2)
                return False
        # Dense
        elif layer[0]==3:
            if not p2 in list(range(g._L_DENSE_UNIT_RANGE[0], g._L_DENSE_UNIT_RANGE[1]+1)):
                if debug:
                    print("Dense unit_width out if range:", p2)
                return False
        # Dropout
        elif layer[0]==4:
            if p3 > g._L_DROUPOUT_RANGE[1] or p2 < g._L_DROUPOUT_RANGE[0] :
                if debug:
                    print("Dropout out if range:", p3)
                return False

    return True
