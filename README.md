# UTokyoMLIntro_GeneticNNOptimizer


<!--- COMMENT TOC generator marker start -->
- [UTokyoMLIntro_GeneticNNOptimizer](#utokyomlintro_geneticnnoptimizer)
  * [Contents:](#contents)
     * [train_model.py](#train_modelpy)
     * [evaluate_model.py](#evaluate_modelpy)
     * [ModelHelper](#modelhelper)
        * [chromosome_helper.py](#chromosome_helperpy)
        * [model.py](#modelpy)
        * [evolution.py](#evolutionpy)
        * [task_queue.py](#task_queuepy)
        * [chromosome_buffer](#chromosome_buffer)
        * [globals.py](#globalspy)
  * [Chromosome Definition](#chromosome-definition)
     * [Definition dictionary](#definition-dictionary)
     * [Layer type and parameters:](#layer-type-and-parameters)
  * [Training Parameters:](#training-parameters)
  * [Evaluation Parameters:](#evaluation-parameters)

<!--- COMMENT TOC generator marker end -->

Neural Network (NN) Model architecture optimization using Genetic Algorithm

*by qlin960618 (Quentin Lin)*

## Contents:

### [train_model.py](./train_model.py)

This is the main script to run for training the model. Parameter of which will be describe below. Current implementation uses the MNIST fashion dataset and with parameters described in the script.

The handling of the generation to generation evolution is done here with only few basic helper functions given in [ModelHelper](./ModelHelper).

**Note:** Use of GPU/Accelerator is currently disabled. As the parallelized version of the chromosome evaluation doesn't really handle device selection very well. All processing is forced on CPU.

### [evaluate_model.py](./evaluate_model.py)

This script will evaluate the training result from the checkpoint of the training script. It takes the last best chromosome of the last generation and evaluate it normally like conventionally how NN model architecture is evaluated.

### [ModelHelper](./ModelHelper)

#### [chromosome_helper.py](./ModelHelper/chromosome_helper.py)

This library provide some functions for managing generation, mutation and error-checking of the chromosome. It import and uses the parameter defined under [globals.py](./ModelHelper/globals.py).

- generate_chromosome()

Generate and return a chromosome at random.

- mutate_chromosome(*parent*)

Takes single chromosome as parent, and return single chromosome to provide. The current implementation only generate single mutation per function call.

- check_chromosome(*chromosome*)

Check the validity of the chromosome. This is useful to ensure that the previously mentioned two function return a valid NN model. However, although parameter might be correct, there is no guarantee that the model is compilable. For this reason, additional error handling in the chromosome evaluation is needed.

#### [model.py](./ModelHelper/model.py)

This library provide the `ChromosomeExpressedModel` class to build, train and test the given chromosome defined NN model. Two version of the Class are provided here. The *Serial* version build and train the TensorFlow NN model  conventionally. The *Parallel* version is the one mostly used here. This version is constructed under python multiprocessing library for parallel testing of the chromosomes.

Detail description of the functions will be abbreviated. But an example function called order are as follow for the *Serial* version

`__constructor__()` -->  `set_train_data()` --> `split_train_test()`--> `build_model()` --> `test_model()`

Model will follow roughly the below structure:

| layer #      | Layer type              |
| ------------ | ----------------------- |
| --           | tf.keras.Input          |
| 0            | *2D layer*              |
| ...          | ...                     |
| `1d_start`-1 | *2D layer*              |
| --           | tf.keras.layers.Flatten |
| `1d_start`   | *1D layer*              |
| ...          | ...                     |
| n            | *1D layer*              |
| --           | tf.keras.layers.Dense   |

#### [evolution.py](./ModelHelper/evolution.py)

This library provide the `EvolutionManager`class to manage mostly the performance evaluation of individual chromosome in the population. `Parallel` or `Serial` version of the chromosome evaluator will be used depending on the `_run_serial` parameter. This class also uses the `ChromosomeBuffer` class for improving performance by avoiding repeat testing. The only main function used here is `process_population_fitness()` that takes in the population and return the corresponding scores. Most of the evolution work-flow is still handle by the main training script.

#### [task_queue.py](./ModelHelper/task_queue.py)

This library and the `TaskQueueManager` is only used when parallel chromosome benchmarking is specified. it takes in a list of jobs and number of parallel workers. When the function `start_and_wait()`is called, it block and wait for all the jobs to complete before return. (returning of result need to be handled separately.)

#### [chromosome_buffer](./ModelHelper/chromosome_buffer)

This library provide the `ChromosomeBuffer`class for managing chromosome performance history. This can help the training process avoid the much longer repeat testing of same chromosome.

#### [globals.py](./ModelHelper/globals.py)

This script provide the parameters to define the possible NN model variation range. Description as follow:

***modifiable parameters***

- _CHROMO_LEN_RANGE_min:

Gives the model's **min** number of 1D+2D layers.

- _CHROMO_LEN_RANGE_max:

Gives the model's **max** number of 1D+2D layers.

- _CHROMO_ACTIVATE_FNC_TYPE:

Define the possible activation function for the `Conv2D` and `Dense` layers to use.

*Free to add or remove additional type*

- _CHROMO_OPTMIZER_TYPE:

Define the possible optimizer that will use in training

*Free to add or remove additional type*

- _MUTATE_OPTMIZER_PROB:

Probably of that in `mutate_chromosome()` the mutation will be on the optimizer

- _LAYER_LEN_CHANGE_PROB:

Probably of that in `mutate_chromosome()` the mutation will be on changing the # of layers.

- _L_CONV2D_KERNEL_RANGE:

The range of the kernel size in `Conv2D` layer.

`kernel_size = (n, n)`

- _L_CONV2D_FILTER_RANGE:

The range of the # of filters in `Conv2D` layer.

`filters = 2**n`

- L_POOLING_SIZE_RANGE:

The range of the pooling size in `MaxPooling2D` and `AveragePooling2D`layers.

`pool_size = (n, n)`

- _L_DENSE_UNIT_RANGE:

The range of the # units that the `Dense` layer can have.

`units = 2**n`

- _L_DROUPOUT_RANGE:

The range of the probability that the `Droupout` layer can take on.

***non-modifiable parameters***

modification of which will also change the [chromosome_helper.py](./ModelHelper/chromosome_helper.py) library

- _CHROMO_2D_LAYER_TYPE:

The possible types of 2D layer that are used in building the model

- _CHROMO_1D_LAYER_TYPE:

The possible types of 1D layer that are used in building the model

- __CHROMO_LAYER_TYPE:

just a variable that combine the above mentioned two.

## Chromosome Definition

### Definition dictionary

```
layers:  [[layer_type, p1, p2, p3], ...(n times)]
opt:    optimizer
1d_start: layer# of first 1d layer
```

### Layer type and parameters:

| Type   | Conv2D         | AveragePooling2D | MaxPooling2D | Dense          | Droupout    |
| ------ | -------------- | ---------------- | ------------ | -------------- | ----------- |
| param1 | Activation Fuc | *d/c*            | *d/c*        | Activation Fuc | *d/c*       |
| param2 | # filters      | pool size        | pool size    | # units        | *d/c*       |
| param3 | kernel size    | *d/c*            | *d/c*        | *d/c*          | probability |

## Training Parameters:

- _SEED:

Random see to use for reproducibility

- POPULATION_HISTORY_PATH:

Path variable passed to `ChromosomeBuffer` for saving chromosome logs

- CHECKPOINT_SAVE_PATH:

Path for saving training checkpoint.

- _train_test_ratio:

ratio of passed in data that will be used for training or testing in chromosome performance evaluation.

- _population_size:

Population size.

- _n_generation:

Number of generation to run

- _mutate_size:

number of chromosome to be mutated in each generation.

**note:** Larger number will result in larger portion of the population to be replaced in each generation.

- _test_epoch_depth:

Number of epoch to used in chromosome performance evaluation.

**note:** This parameter perhaps a difficult to tune. As the effect of high value will result in the convergence of NN model test result and a much longer training time. But a low value will makes the process favor NN model with less tunable parameters and layers.

- _mp_worker_size:

Number of parallel workers to run in chromosome performance evaluation.

## Evaluation Parameters:

- POPULATION_HISTORY_PATH:

Set to something not `None` or else will be loaded from [train_model.py](./train_model.py).

- CHECKPOINT_SAVE_PATH:

Set to something not `None` or else will be loaded from [train_model.py](./train_model.py).

- _train_epochs:

Number of epochs to used for training
