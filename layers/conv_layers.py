from . import OPERATOR
import numpy as np
import tensorflow as tf
from tensorflow import keras

@OPERATOR.register_operator("BatchNormalization")
class TFBatchNormalization(keras.layers.Layer):
    def __init__(self, weight, bias, running_mean, running_var, epsilon=1e-5, momentum=0.9):
        super().__init__()
        epsilon = 1e-5 if epsilon is None else epsilon
        momentum = 0.9 if epsilon is None else momentum
        self.bn = keras.layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(bias),
            gamma_initializer=keras.initializers.Constant(weight),
            moving_mean_initializer=keras.initializers.Constant(running_mean),
            moving_variance_initializer=keras.initializers.Constant(running_var),
            epsilon=epsilon,
            momentum=momentum)

    def call(self, inputs):
        return self.bn(inputs)