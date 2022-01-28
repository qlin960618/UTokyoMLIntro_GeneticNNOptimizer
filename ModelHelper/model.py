import os, sys
import multiprocessing as mp
# from tqdm.contrib.concurrent import process_map  # or thread_map

import numpy as np
import math, random
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#### too much headach running multiple job on GPU so...
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from sklearn.model_selection import train_test_split

import ModelHelper.globals as g

RAN_SEED = 0


"""
model structure
---------------------
...                     layer=0
~~~~~~~~~
...                     layer=d1_start-1
---------------------
tf.keras.layers.Flatten()
---------------------
...                     layer=d1_start
~~~~~~~~~
...                     layer=n
---------------------
tf.keras.layers.Dense(output_shape, activation=output_activation)
---------------------
"""
class ChromosomeExpressedModel_Serial():
    def __init__(self, chromosome, seed=RAN_SEED):
        self.chromosome = chromosome
        self.incompatable_model=True
        self.result=0

        #seed random generator
        # random.seed(seed)
        tf.random.set_seed(seed)
        self.seed = seed+1
        # np.random.seed(seed)

    def set_train_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    def set_test_data(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    ## preform train test split internally
    def split_train_test(self, train_ratio):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train, self.y_train,
                                                    test_size=train_ratio,
                                                    random_state=self.seed)


    def build_model(self, input_shape, output_shape, output_activation, validate_only=False):

        layers=[]
        print("input shape:", input_shape)
        layers.append(tf.keras.Input(shape=input_shape))

        d1_s = self.chromosome["1d_start"]

        for ind, layer_params in enumerate(self.chromosome['layers']):
            layer_type, p1, p2, p3= layer_params
            act_fnc = g._CHROMO_ACTIVATE_FNC_TYPE[p1]
            # print(act_fnc)
            #build 2d lauer
            if ind < d1_s:
                if layer_type == 0:
                    if p1!=0:
                        layers.append(tf.keras.layers.Conv2D(2**p2,
                                                kernel_size=(p3, p3),
                                                activation=act_fnc))
                    else:
                        layers.append(tf.keras.layers.Conv2D(2**p2,
                                                kernel_size=(p3, p3)))
                if layer_type == 1:
                    layers.append(tf.keras.layers.AveragePooling2D(pool_size=(p2, p2)))
                if layer_type == 2:
                    layers.append(tf.keras.layers.MaxPooling2D(pool_size=(p2, p2)))
            if ind==d1_s:
                layers.append(tf.keras.layers.Flatten())
            if ind>=d1_s:
                if layer_type == 3:
                    if p1!=0:
                        layers.append(tf.keras.layers.Dense(2**p2,
                                                activation=act_fnc))
                    else:
                        layers.append(tf.keras.layers.Dense(2**p2))
                if layer_type == 4:
                    layers.append(tf.keras.layers.Dropout(p3))

        # no 1D layer in definition
        if d1_s==len(self.chromosome['layers']):
            layers.append(tf.keras.layers.Flatten())

        # Last output layer
        layers.append(tf.keras.layers.Dense(output_shape,activation=output_activation))

        try:
            self.model = tf.keras.models.Sequential(layers)
            self.model.compile(optimizer=self.chromosome["opt"],
                            loss="sparse_categorical_crossentropy",
                            metrics=['accuracy'])

        except Exception as e:
            print("Model failled: ", e)
            return False

        self.incompatable_model=False
        return True

    def test_model(self, _epoch):
        if self.incompatable_model:
            return 0

        try:
            self.model.summary()
            self.model.fit(self.x_train, self.y_train, epochs=_epoch)

            result = self.model.evaluate(self.x_test, self.y_test)

            return result[1]
        except Exception as e:
            print("Training failled: ", e)
            return 0

class ChromosomeExpressedModel_Parallel(mp.Process):
    def __init__(self, chromosome, seed=RAN_SEED):
        super(ChromosomeExpressedModel_Parallel, self).__init__()
        self.chromosome = chromosome
        self.incompatable_model=True
        self.model_failled=False

        #seed random generator
        # random.seed(seed)
        tf.random.set_seed(seed)
        self.seed = seed+1

        # np.random.seed(seed)
    def set_train_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    def set_test_data(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test


    ## preform train test split internally
    def split_train_test(self, train_ratio):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train, self.y_train,
                                                    test_size=train_ratio,
                                                    random_state=self.seed)

    def supress_stdout(self):
        sys.stdout = open(os.devnull, 'w')


    def build_model(self, input_shape=None, validate_only=False):
        if input_shape is None:
            input_shape=self.x_train.shape[1:]
        layers=[]
        # print("input shape:", input_shape)
        layers.append(tf.keras.Input(shape=input_shape))

        d1_s = self.chromosome["1d_start"]

        for ind, layer_params in enumerate(self.chromosome['layers']):
            layer_type, p1, p2, p3= layer_params
            act_fnc = g._CHROMO_ACTIVATE_FNC_TYPE[p1]
            # print(act_fnc)
            #build 2d lauer
            if ind < d1_s:
                if layer_type == 0:
                    if p1!=0:
                        layers.append(tf.keras.layers.Conv2D(2**p2,
                                                kernel_size=(p3, p3),
                                                activation=act_fnc))
                    else:
                        layers.append(tf.keras.layers.Conv2D(2**p2,
                                                kernel_size=(p3, p3)))
                if layer_type == 1:
                    layers.append(tf.keras.layers.AveragePooling2D(pool_size=(p2, p2)))
                if layer_type == 2:
                    layers.append(tf.keras.layers.MaxPooling2D(pool_size=(p2, p2)))
            if ind==d1_s:
                layers.append(tf.keras.layers.Flatten())
            if ind>=d1_s:
                if layer_type == 3:
                    if p1!=0:
                        layers.append(tf.keras.layers.Dense(2**p2,
                                                activation=act_fnc))
                    else:
                        layers.append(tf.keras.layers.Dense(2**p2))
                if layer_type == 4:
                    layers.append(tf.keras.layers.Dropout(p3))
        # no 1D layer in definition
        if d1_s==len(self.chromosome['layers']):
            layers.append(tf.keras.layers.Flatten())

        # Last output layer
        # print("sdadasdasasdsads:", self.output_shape, self.output_activation)
        layers.append(tf.keras.layers.Dense(self.output_shape,activation=self.output_activation))

        try:
            self.model = tf.keras.models.Sequential(layers)
            self.model.compile(optimizer=self.chromosome["opt"],
                            loss="sparse_categorical_crossentropy",
                            metrics=['accuracy'])

        except Exception as e:
            # print("Model failled: ", e)
            self.model_failled=True
            return False

        self.incompatable_model=False
        return True

    def test_model(self, _epoch):
        if self.incompatable_model:
            return 0

        try:
            # self.model.summary()
            tqdm_text = "job#" + "{}".format(self.job_id).zfill(3)

            with tqdm(total=_epoch+1, desc=tqdm_text, position=self.index+1) as pbar:
                self.model.fit(self.x_train, self.y_train,
                                epochs=_epoch,
                                callbacks=[ProgressCallback(pbar)],
                                verbose=0)

                result = self.model.evaluate(self.x_test, self.y_test,
                                            verbose=0)
                pbar.update(1)


            self.result = result[1]
            return result[1]
        except Exception as e:
            self.model_failled=True
            self.result = 0
            return 0

    def set_run_args(self, output_shape,
                        output_activation,
                        epoch_depth,
                        queue):
        # print("set_run_arg: ", output_shape)
        # print("set_run_arg: ", output_activation)
        # print("set_run_arg: ", epoch_depth)
        # print("set_run_arg: ", queue)

        self.output_shape=output_shape
        self.output_activation=output_activation
        self.epoch_depth=epoch_depth
        self.queue = queue

        # self.initargs=(mp.RLock(),)
        # self.initializer=tqdm.set_lock

    def set_run_index(self, index, job_id):
        self.index=index
        self.job_id=job_id

    def run(self):
    # def run(self, epoch_depth):
        # self.supress_stdout()
        tqdm.set_lock(mp.RLock())

        self.build_model(input_shape=self.x_train.shape[1:])
        self.test_model(self.epoch_depth)

        if not self.model_failled:
            self.queue.put(self.result)
        else:
            self.queue.put(0)

class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, pbar):
        self.pbar=pbar
    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)
        # self.pbar.set_postfix_str()
###
