
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import math, random


from ModelHelper.chromosome_buffer import ChromosomeBuffer

## set to not None to disable loading
POPULATION_HISTORY_PATH=None
CHECKPOINT_SAVE_PATH=None

if POPULATION_HISTORY_PATH is None:
    from train_model import POPULATION_HISTORY_PATH
if CHECKPOINT_SAVE_PATH is None:
    from train_model import CHECKPOINT_SAVE_PATH



FIGURE_SAVE_PATH="./Figures"



def main():
    ###############################################
    #### Load data
    ###############################################
    data=load_data_log(CHECKPOINT_SAVE_PATH)
    buffer=ChromosomeBuffer(POPULATION_HISTORY_PATH)


    ###############################################
    #### preprocessing
    ###############################################
    fitness_history = np.array(data['fitness_history'])
    n_generation = fitness_history.shape[0]
    current_population = data['population']
    current_fitness = fitness_history[-1, :]

    print("Data to plot:",fitness_history.shape)

    population_size=fitness_history.shape[1]
    gen_list = np.array(list(range(n_generation)))


    ###############################################
    #### plot score of each chromosome over generation
    ###############################################
    fig, ax = plt.subplots(1,1, figsize=(8,4))
    ax.set_xlabel("Generation [-]")
    ax.set_ylabel("Score [-]")
    ax.set_title("Generation score history")
    # for chromo_id in range(5):
    for chromo_id in range(population_size):

        rescaled = fitness_history[:, chromo_id]
        # rescaled = np.log(fitness_history[:, chromo_id])
        ax.plot(gen_list, rescaled)

    # ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_SAVE_PATH,
                "all_population.png"))

    ###############################################
    #### plot mean score of population over generation
    ###############################################
    population_mean=np.mean(fitness_history, axis=1)
    print(population_mean.shape)
    fig, ax = plt.subplots(1,1, figsize=(8,4))
    ax.set_xlabel("Generation [-]")
    ax.set_ylabel("Mean Score [-]")
    ax.set_title("Generation mean score history")

    ax.plot(gen_list, population_mean)

    # ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_SAVE_PATH,
                "population_mean.png"))




    plt.show()

def save_data_log(path, data):
    pickle.dump( data, open( path, "wb" ) )


def load_data_log(path):
    return pickle.load( open( path, "rb" ) )

if __name__=='__main__':
    main()
