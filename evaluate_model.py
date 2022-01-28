import os, sys

# use GPU if there is one
os.environ['MODEL_RUN_EVALUATE']=""
from ModelHelper.model import ChromosomeExpressedModel_Parallel
from ModelHelper.chromosome_buffer import ChromosomeBuffer

import tensorflow.keras.datasets as tf_datasets

import numpy as np
import math, random
import pickle

## set to not None to disable loading
POPULATION_HISTORY_PATH=None
CHECKPOINT_SAVE_PATH=None

if POPULATION_HISTORY_PATH is None:
    from train_model import POPULATION_HISTORY_PATH
if CHECKPOINT_SAVE_PATH is None:
    from train_model import CHECKPOINT_SAVE_PATH


##
_train_epochs=10



def main():
    data=load_data_log(CHECKPOINT_SAVE_PATH)
    buffer=ChromosomeBuffer(POPULATION_HISTORY_PATH)
    # print(data)
    fitness_history = np.array(data['fitness_history'])
    generation = fitness_history.shape[0]
    current_population = data['population']
    current_fitness = fitness_history[-1, :]



    fitness_sort_arg = np.argsort(current_fitness)
    best_fit = fitness_sort_arg[-1]
    print("current best fit model is: ", best_fit, " at generation:", generation,
            " with fitness: ",current_fitness[best_fit])
    print("with parameters: ", current_population[best_fit])

    ## load dataset
    (x_train, y_train), (x_val, y_val) = tf_datasets.fashion_mnist.load_data()
    x_train, x_val = x_train/255.0, x_val/255.0

    x_train=np.expand_dims(x_train, axis=3)
    x_val=np.expand_dims(x_val, axis=3)


    result = run_evaluation_on_chromosome(current_population[best_fit],
                                        x_train, y_train, x_val, y_val)

    print("Best of current generation:", result)


def run_evaluation_on_chromosome(chromosome, x_train, y_train, x_val, y_val):
    ## run evaluation on the best model
    # well... Parallel version is more up to date, might as well just use it and run it serially
    model = ChromosomeExpressedModel_Parallel(chromosome)
    model.set_train_data(x_train=x_train, y_train=y_train)
    model.set_test_data(x_test=x_val, y_test=y_val)

    #only output_shape and output_activation is needed to set
    model.set_run_args(output_shape=10, output_activation='softmax',
                        epoch_depth=None,queue=None)

    print("Building model...")
    succ = model.build_model()

    if not succ:
        print("build failled")
        raise RuntimeError("Model Build and Compile Failled")

    # Run train and test
    result = model.test_model(_epoch=_train_epochs, on_train=False)
    return result


def save_data_log(path, data):
    pickle.dump( data, open( path, "wb" ) )


def load_data_log(path):
    return pickle.load( open( path, "rb" ) )

if __name__=='__main__':
    main()
