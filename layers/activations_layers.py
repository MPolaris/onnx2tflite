import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import OPERATOR
from .common_layers import TFPad

@OPERATOR.register_operator("Relu")
class TFRelu(keras.layers.Layer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def call(self, inputs):
        return keras.activations.relu(inputs)

@OPERATOR.register_operator("Sigmoid")
class TFSigmoid(keras.layers.Layer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def call(self, inputs):
        return keras.activations.relu(inputs)

@OPERATOR.register_operator("LeakyRelu")
class TFLeakyRelu(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        self.alpha = node_attribute['alpha']

    def call(self, inputs):
        return keras.activations.relu(inputs, alpha=self.alpha)

@OPERATOR.register_operator("Tanh")
class TFTanh(keras.layers.Layer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def call(self, inputs):
        return keras.activations.tanh(inputs)