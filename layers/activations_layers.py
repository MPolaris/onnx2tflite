from tensorflow import keras

from .shape_axis_utils import Torch2TFAxis
from . import OPERATOR

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

@OPERATOR.register_operator("Softmax")
class TFSoftmax():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        self.axis = Torch2TFAxis(node_attribute['axis'])

    def __call__(self, inputs):
        return keras.activations.softmax(inputs, axis=self.axis)

@OPERATOR.register_operator("Selu")
class TFSelu():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return keras.activations.selu(inputs)