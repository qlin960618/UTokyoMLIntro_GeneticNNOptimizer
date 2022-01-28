#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np


# In[2]:


# random.seed(100)

target_sentence = "Hello! How are you?"

gene_pool = [32, 33, 63]+[i+97 for i in range(23)]+[i+65 for i in range(23)]
gene_pool = "".join([chr(i) for i in gene_pool])

population_size=50


# In[3]:


def generate_chromosome(length):
    genes=[]
    while len(genes) < length:
        genes.append(gene_pool[random.randrange(0, len(gene_pool))])
    return ''.join(genes)


def calculate_fitness(chromosome):
    fitness = 0
    for i, ch in enumerate(chromosome):
        if ch == target_sentence[i]:
            fitness +=1
    return fitness



def crossover(parent1, parent2, ncross):
    cpoints = random.sample([i for i in range(len(parent1))], ncross)
    cpoints.sort(reverse=False)
    child1 = parent1
    child2 = parent2
    for i in cpoints:
        tmp=child1[i:]
        child1 = child1[:i] + child2[i:]
        child2 = child2[:i] + tmp
    return child1, child2

def mutate(chromosome):
    index_to_mutate = random.randrange(0, len(chromosome))
    gene = list(chromosome)
    mutated_gene = gene_pool[random.randrange(0, len(gene_pool))]
    gene[index_to_mutate]=mutated_gene
    return ''.join(gene)


# In[4]:


population = []

for i in range(population_size):
    population.append(generate_chromosome(len(target_sentence)))

print(population)

population_fitness = []

for chromosome in population:
    population_fitness.append(calculate_fitness(chromosome))

population_fitness = np.array(population_fitness)

print(population_fitness)






# In[5]:


print_interval=5000
ncross = 6

for generation in range(5000000):
    fit_sort_arg = np.argsort(population_fitness)
    parent1_index = fit_sort_arg[-1]
    parent2_index = fit_sort_arg[-2]

#     print(population_fitness)
#     print(parent1_index, parent2_index)
#     print(fit_sort_arg)

    parent1 = population[parent1_index]
    parent2 = population[parent2_index]
    child1, child2 = crossover(parent1, parent2, ncross)
    child1 = mutate(child1)
    child2 = mutate(child2)

#     print("Parent: ", parent)
#     print("Child: ", child)
    child_fitness1 = calculate_fitness(child1)
    child_fitness2 = calculate_fitness(child2)
#     print("Child Fitness: ", child_fitness)


    #remove the lease fit
    index_to_delete = fit_sort_arg[0]

    population_fitness[fit_sort_arg[0]] = child_fitness1
    population[fit_sort_arg[0]] = child1
    population_fitness[fit_sort_arg[1]] = child_fitness2
    population[fit_sort_arg[1]] = child2


    if generation % print_interval==0:
        print("Generation {:d}".format(generation))
        print("Current Population:", population)
        print("Current Fitness:", population_fitness)

    if child1 == target_sentence or child2 == target_sentence:
        print("Solution found at Generation", generation)
        print("Current Population:", population)
        print("Current Fitness:", population_fitness)
        break


print("Evolution end at generation {:d}".format(generation))
print("Current Population:", population)
print("Current Fitness:", population_fitness)


# In[ ]:
