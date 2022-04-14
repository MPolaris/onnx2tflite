import numpy as np
import tensorflow as tf
import logging
from tensorflow import keras

from . import OPERATOR
from . import shape_axis_utils

LOG = logging.getLogger("calculations_layers :")

@OPERATOR.register_operator("Add")
class TFAdd():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.t1 = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[0]])
        self.t2 = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[1]])
        debug = 1

    def __call__(self, *args, **kwargs):
        return self.t1 + self.t2

@OPERATOR.register_operator("Sub")
class TFSub():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.t1 = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[0]])
        self.t2 = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[1]])

    def __call__(self, *args, **kwargs):
        return self.t1 - self.t2

@OPERATOR.register_operator("Mul")
class TFMul():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.t1 = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[0]])
        self.t2 = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[1]])

    def __call__(self, *args, **kwargs):
        return self.t1 * self.t2

@OPERATOR.register_operator("Div")
class TFDiv():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.t1 = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[0]])
        self.t2 = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[1]])

    def __call__(self, *args, **kwargs):
        return self.t1 / self.t2

@OPERATOR.register_operator("Pow")
class TFPow():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.power_index = node_weights[node_inputs[1]]

    def __call__(self, inputs, *args, **kwargs):
        return tf.pow(inputs, self.power_index)

@OPERATOR.register_operator("Reciprocal")
class TFReciprocal():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs, *args, **kwargs):
        return 1/inputs

@OPERATOR.register_operator("Sqrt")
class TFSqrt():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs, *args, **kwargs):
        return tf.sqrt(inputs)
