#Chromoson definition
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

# chromosome length range
_CHROMO_LEN_RANG_min=4
_CHROMO_LEN_RANG_max=8


_CHROMO_ACTIVATE_FNC_TYPE={
    0:"",
    1:"relu",
    2:"tanh",
}

_CHROMO_OPTMIZER_TYPE=[
    "adam",
    "SGD",
]

_MUTATE_OPTMIZER_PROB=0.07
_LAYER_LEN_CHANGE_PROB=0.08

_L_CONV2D_KERNEL_RANGE=(2, 4)   # (n, n)
_L_CONV2D_FILTER_RANGE=(1, 6)   # 2^n layers
_L_POOLING_SIZE_RANGE=(2, 3)    # range for pool size (n, n)
_L_DENSE_UNIT_RANGE=(1, 9)      # 2^n
_L_DROUPOUT_RANGE=(0, 0.9)      # from no droupout to 90%


######################## Not modifiable variable ###############################
_CHROMO_2D_LAYER_TYPE={
    0:'Conv2D',             #tf.keras.layers.Conv2D,
    1:'AveragePooling2D',   #tf.keras.layers.AveragePooling2D,
    2:'MaxPooling2D',       #tf.keras.layers.MaxPooling2D,
}
_CHROMO_1D_LAYER_TYPE={
    3:'Dense',              #tf.keras.layers.Dense,
    4:'Dropout',            #tf.keras.layers.Dropout,
}

_CHROMO_LAYER_TYPE = {**_CHROMO_2D_LAYER_TYPE, **_CHROMO_1D_LAYER_TYPE}
################################################################################
