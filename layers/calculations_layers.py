import logging
import tensorflow as tf
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

@OPERATOR.register_operator("MatMul")
class TFMatMul():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.t1 = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[0]])
        self.t2 = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[1]])

    def __call__(self, *args, **kwargs):
        return tf.matmul(self.t1, self.t2)

@OPERATOR.register_operator("Pow")
class TFPow():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.power_index = node_weights[node_inputs[1]]

    def __call__(self, inputs, *args, **kwargs):
        return tf.pow(inputs, self.power_index)

@OPERATOR.register_operator("Reciprocal")
class TFReciprocal():
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs, *args, **kwargs):
        return 1/inputs

@OPERATOR.register_operator("Sqrt")
class TFSqrt():
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs, *args, **kwargs):
        return tf.sqrt(inputs)

@OPERATOR.register_operator("Exp")
class TFSqrt():
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs, *args, **kwargs):
        return tf.exp(inputs)

@OPERATOR.register_operator("Log")
class TFLog():
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs, *args, **kwargs):
        return tf.log(inputs)

@OPERATOR.register_operator("ReduceMean")
class TFReduceMean():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.keep_dims = node_attribute.get("keepdims", 0) == 1
        input_shape_len = len(tensor_grap[node_inputs[0]].shape)
        if input_shape_len > 2:
            # TODO
            raise NotImplementedError("ReduceMean not implemented when input shape length > 2")
        else:
            self.axis = [shape_axis_utils.Torch2TFAxis(i) if i >=0 else input_shape_len + i for i in node_attribute.get("axes", [-1])]

    def __call__(self, inputs, *args, **kwargs):
        return tf.math.reduce_mean(inputs, axis=self.axis, keepdims=self.keep_dims)

@OPERATOR.register_operator("ArgMax")
class TFArgMax():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.axis = shape_axis_utils.Torch2TFAxis(node_attribute['axis'])
        self.keepdims = node_attribute.get("keepdims", 0) == 1

    def __call__(self, inputs, *args, **kwargs):
        _inputs = tf.argmax(inputs, axis=self.axis)
        if self.keepdims:
            _inputs = tf.expand_dims(_inputs, axis=self.axis)
        return _inputs

@OPERATOR.register_operator("ArgMin")
class TFArgMin():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.axis = shape_axis_utils.Torch2TFAxis(node_attribute['axis'])
        self.keepdims = node_attribute.get("keepdims", 0) == 1

    def __call__(self, inputs, *args, **kwargs):
        _inputs = tf.argmax(inputs, axis=self.axis)
        if self.keepdims:
            _inputs = tf.expand_dims(_inputs, axis=self.axis)
        return _inputs