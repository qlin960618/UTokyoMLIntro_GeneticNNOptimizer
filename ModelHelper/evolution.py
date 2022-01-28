import numpy as np
import math
import time

import multiprocessing as mp
from tqdm import tqdm

from ModelHelper.model import ChromosomeExpressedModel_Serial, ChromosomeExpressedModel_Parallel
from ModelHelper.task_queue import TaskQueueManager
from .chromosome_buffer import ChromosomeBuffer

import tensorflow as tf




_verbose=False
_run_serial=False
_use_task_queue=True

class EvolutionManager():
    def __init__(self, epoch_depth, mp_worker_size, population_size):
        self.epoch_depth = epoch_depth
        self.n_workers = mp_worker_size
        self.population_size = population_size
        self.population_buffer = None
        self.generation_cnt=0

    def set_train_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def set_test_data(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def load_history(self, csvpath):
        self.population_buffer = ChromosomeBuffer(csvpath, verbose=_verbose)


    def run_worker(self, worker):
        worker.run(10, 'softmax', self.epoch_depth)
        # worker.run(self.epoch_depth)
    def set_train_test_ratio(self, ratio):
        self.train_test_ratio=ratio

    def process_population_fitness(self, population):
        n_chromosomes = len(population)
        fitness_out = [0]*n_chromosomes
        self.generation_cnt+=1

        # skip simulation if exist in population memory
        to_run_buf = []
        to_run_buf_ind = []
        to_run_buf_score = []
        for ind, chromosome in enumerate(population):
            # print("sdfsdfsdfs, ", ind, chromosome )
            existed, score = self.population_buffer.match_entries(chromosome)
            if existed:
                fitness_out[ind]=score
            else:
                to_run_buf.append(chromosome)
                to_run_buf_ind.append(ind)
                to_run_buf_score.append(0)
        #######################################################
        # run simulation for chromosome in to_run_buf
        if len(to_run_buf)>0:
            print("Running generation: {:d} -- {:d} need simulation".format(self.generation_cnt, len(to_run_buf_ind)))
            if _run_serial:
                for i in range(len(to_run_buf_score)):
                    print("Running Cromosome: {:d}".format(i))
                    print("params: {}".format(str(to_run_buf[i])))
                    model=ChromosomeExpressedModel_Serial(to_run_buf[i],
                                                    seed=self.generation_cnt*self.population_size+i)
                    model.set_train_data(self.x_train, self.y_train)
                    model.split_train_test(self.train_test_ratio)
                    # model.set_test_data(self.x_test, self.y_test)
                    # succ = model.build_model(tuple([*self.x_train.shape[1:], 1]),
                    succ = model.build_model(self.x_train.shape[1:],
                                        10, 'softmax')
                    if not succ:
                        to_run_buf_score[i]=0
                        continue
                    else:
                        score = model.test_model(self.epoch_depth)
                        to_run_buf_score[i]=score
            #run parallel using multiprocessing
            else:
                jobs=[]
                jobs_queue=[]
                n_jobs = len(to_run_buf_score)

                # tqdm.set_lock(mp.RLock())


                for i in range(n_jobs):
                    model=ChromosomeExpressedModel_Parallel(to_run_buf[i],
                                                    seed=self.generation_cnt+i)
                    model.set_train_data(self.x_train, self.y_train)
                    model.split_train_test(self.train_test_ratio)
                    # model.set_test_data(self.x_test, self.y_test)
                    jobs_queue.append(mp.Queue())
                    model.set_run_args(output_shape=10,
                                        output_activation='softmax',
                                        epoch_depth=self.epoch_depth,
                                        queue=jobs_queue[i])
                    # model.build_model()
                    jobs.append(model)

                print("Generation: {:d} Initialier done, Launching Task Queue...".format(self.generation_cnt))

                if _use_task_queue:
                    #this is causing too much headach rn....
                    task_queue_manager = TaskQueueManager(n_workers=self.n_workers,
                                                        jobs=jobs)

                    task_queue_manager.start_and_wait()
                else:
                    for i in range(n_jobs):
                        jobs[i].set_run_index(index=i, job_id=i)
                        jobs[i].start()
                    for i in range(n_jobs):
                        jobs[i].join()
                print("Generation: {:d} Done, getting result".format(self.generation_cnt))
                for i in range(n_jobs):
                    to_run_buf_score[i] = jobs_queue[i].get()
                # with mp.Pool(self.n_workers) as mp_p:
                #     to_run_buf_score = mp_p.map(self.run_worker, workers)


        else:
            print("No unique chromosome in generation: {:d}".format(self.generation_cnt))

        #######################################################


        ## add simulation result to population buffer and output
        for ind, gind in enumerate(to_run_buf_ind):
            fitness_out[gind]=to_run_buf_score[ind]
            self.population_buffer.add_entry(population[gind], to_run_buf_score[ind])

        return fitness_out
