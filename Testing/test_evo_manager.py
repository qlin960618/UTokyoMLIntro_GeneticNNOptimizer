from ModelHelper import *


import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import math, random

import multiprocessing as mp


_SEED=14
_validation_ratio = 0.5

#model history
POPULATION_HISTORY_PATH="./history/population_history.csv"


_N_GENERATION = 5
_MUTATE_SIZE=4

_TEST_EPOCH_DEPTH=2

_mp_batch_size=8
population_size=8







def main():


    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0

    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                test_size=1-_validation_ratio,
                                                random_state=_SEED)

    evoManager = EvolutionManager(epoch_depth=_TEST_EPOCH_DEPTH,
                                mp_batch_size=_mp_batch_size)
    evoManager.load_history(POPULATION_HISTORY_PATH)

    print("train:", x_train.shape)
    print("test:", x_test.shape)
    print("val:", x_val.shape)

    evoManager.set_train_data(x_train=x_train, y_train=y_train)
    evoManager.set_test_data(x_test=x_test, y_test=y_test)


    print("generate initial population")
    population = []
    random.seed(_SEED)
    for i in range(population_size):
        population.append(generate_chromosome())
        print(population[i])
    # print(population)


    population_fitness = evoManager.process_population_fitness(population)

    print(population_fitness)

    exit()
    fit_sort_arg = np.argsort(population_fitness)
    parents_index = fit_sort_arg[::-1][:_MUTATE_SIZE]




if __name__=='__main__':
    ## windows mp support required
    mp.freeze_support()

    main()
