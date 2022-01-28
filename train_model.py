from ModelHelper import *


import tensorflow.keras.datasets as tf_datasets
# from sklearn.model_selection import train_test_split
import numpy as np
import math, random
# import scipy.io as scio
import pickle, copy

import multiprocessing as mp


_SEED=12
_train_test_ratio = 0.8


Trial=1
#model history
POPULATION_HISTORY_PATH="./history/{:2d}_population_history.csv".format(Trial)
CHECKPOINT_SAVE_PATH="./history/{:2d}_run_checkpoint.p".format(Trial)

# _n_generation = 30
_n_generation = 500
_mutate_size=12

_test_epoch_depth=3

_mp_worker_size=3
# _population_size=16
_population_size=30

print_interval=1
save_interval=1



"""
## filler function
"""
def process_population_fitness(population,
                test_epoch_depth,
                mp_batch_size,
                x_train, y_train,
                x_val, y_val):
    return [0 for i in  range(len(population))]


def main():
    ###################
    ## Load dataset
    ###################
    # fashion_mnist = tf_datasets.fashion_mnist
    (x_train, y_train), (x_val, y_val) = tf_datasets.fashion_mnist.load_data()
    x_train, x_val = x_train/255.0, x_val/255.0

    x_train=np.expand_dims(x_train, axis=3)
    x_val=np.expand_dims(x_val, axis=3)

    # x_val, x_val, y_val, y_val = train_test_split(x_val, y_val,
    #                                             test_size=1-_validation_ratio,
    #                                             random_state=_SEED)
    ###################
    ## create Evolution Manager Class
    ###################
    evoManager = EvolutionManager(epoch_depth = _test_epoch_depth,
                                mp_worker_size = _mp_worker_size,
                                population_size = _population_size)
    evoManager.load_history(POPULATION_HISTORY_PATH)

    print("train:", x_train.shape)
    print("val:", x_val.shape)
    print("train test ratio:", _train_test_ratio)
    # print("val:", x_val.shape)

    evoManager.set_train_data(x_train=x_train, y_train=y_train)
    evoManager.set_train_test_ratio(ratio=_train_test_ratio)
    ## preform train test split internally
    # evoManager.set_test_data(x_val=x_val, y_val=y_val)

    #load from checkpoit if exist
    if not os.path.isfile(CHECKPOINT_SAVE_PATH):
        print("generate initial population")
        population = []
        population_fitness_history=[]
        gene_init=0
        random.seed(_SEED)
        for i in range(_population_size):
            population.append(generate_chromosome())
        print(population)
    else:
        print("checkpoint exist, loading from save")
        data=load_data_log(CHECKPOINT_SAVE_PATH)
        population_fitness_history = data['fitness_history']
        gene_init = data['generation']
        population = data['population']
        evoManager.generation_cnt=gene_init



    print("iterate generation")
    for generation in range(gene_init, _n_generation+gene_init):
        random.seed(_SEED+generation)
        population_fitness = evoManager.process_population_fitness(population)
        last_population=copy.deepcopy(population)

        fit_sort_arg = np.argsort(population_fitness)
        fit_index = fit_sort_arg[::-1][:_mutate_size]
        unfit_index = fit_sort_arg[:_mutate_size]

        #mutate the best fits
        childs=[]
        for parent_ind in fit_index:
            child = mutate_chromosome(population[parent_ind])
            childs.append(child)

        # remove the least fit
        for ind, ind_drop in enumerate(unfit_index):
            population[ind_drop]=childs[ind]


        population_fitness_history.append(population_fitness)
        if generation % print_interval==0:
            print("Current Fitness:", population_fitness)

        if generation % save_interval==0:
            save_data={
                'fitness_history':population_fitness_history,
                'generation':generation,
                'population':last_population,
            }
            save_data_log(CHECKPOINT_SAVE_PATH, save_data)

    print("Evolution end at generation {:d}".format(generation))
    print("Current Population:", population)
    print("Current Fitness:", population_fitness)

    save_data={
        'fitness_history':population_fitness_history,
        'generation':generation,
        'population':last_population,
    }
    save_data_log(CHECKPOINT_SAVE_PATH, save_data)

def save_data_log(path, data):
    pickle.dump( data, open( path, "wb" ) )


def load_data_log(path):
    return pickle.load( open( path, "rb" ) )

if __name__=='__main__':
    ## windows mp support required
    mp.freeze_support()

    main()
