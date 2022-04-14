import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import OPERATOR
from .common_layers import TFPad

@OPERATOR.register_operator("Relu")
class TFRelu():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return keras.activations.relu(inputs)

@OPERATOR.register_operator("Sigmoid")
class TFSigmoid():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return keras.activations.sigmoid(inputs)

@OPERATOR.register_operator("LeakyRelu")
class TFLeakyRelu():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        self.alpha = node_attribute['alpha']

    def __call__(self, inputs):
        return keras.activations.relu(inputs, alpha=self.alpha)

@OPERATOR.register_operator("Tanh")
class TFTanh():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return keras.activations.tanh(inputs)