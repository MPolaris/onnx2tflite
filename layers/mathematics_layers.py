import logging
import numpy as np
import tensorflow as tf

from utils.op_registry import OPERATOR
from layers import dimension_utils

LOG = logging.getLogger("calculations_layers :")

def np2tf(x):
    if isinstance(x, np.ndarray):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return x, False
    return x, True

def match_tensor(x1:tf.Tensor or np.ndarray, x2:tf.Tensor or np.ndarray):
    
    x1, f1 = np2tf(x1)
    x2, f2 = np2tf(x2)

    # no need to transpose if all var are tensor, we assume tensor are computed by gragh.
    if f1 and f2:
        return x1, x2
    
    # ensure tensor is set to x1, weights set to x2
    if f2:
        x1, x2 = x2, x1

    if x1.shape.ndims != x2.shape.ndims:
        while x2.shape.ndims < x1.shape.ndims:
            x2 = tf.expand_dims(x2, axis=0)
    
    new_shape = dimension_utils.shape_NCD_to_NDC_format([i for i in range(len(x2.shape))])
    x2 = tf.transpose(x2, new_shape)
    return (x2, x1) if f2 else (x1, x2)

@OPERATOR.register_operator("Add")
class TFAdd():
    def __init__(self, tensor_grap, node_weights, node_inputs, *args, **kwargs):
        super().__init__()
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)

    def __call__(self, *args, **kwargs):
        return self.first_operand + self.second_operand

@OPERATOR.register_operator("Sub")
class TFSub():
    def __init__(self, tensor_grap, node_weights, node_inputs, *args, **kwargs):
        super().__init__()
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)

    def __call__(self, *args, **kwargs):
        return self.first_operand - self.second_operand

@OPERATOR.register_operator("Mul")
class TFMul():
    def __init__(self, tensor_grap, node_weights, node_inputs, *args, **kwargs):
        super().__init__()
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)

    def __call__(self, *args, **kwargs):
        return self.first_operand * self.second_operand

@OPERATOR.register_operator("Div")
class TFDiv():
    def __init__(self, tensor_grap, node_weights, node_inputs, *args, **kwargs):
        super().__init__()
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)

    def __call__(self, *args, **kwargs):
        return self.first_operand / self.second_operand

@OPERATOR.register_operator("MatMul")
class TFMatMul():
    def __init__(self, tensor_grap, node_weights, node_inputs, *args, **kwargs):
        super().__init__()
        if node_inputs[0] in tensor_grap:
            self.first_operand = tensor_grap[node_inputs[0]]
            new_shape = [0, self.first_operand.shape.ndims-1] + [i for i in range(1, self.first_operand.shape.ndims-1)]
            self.first_operand = tf.transpose(self.first_operand, perm=new_shape)
        else:
            self.first_operand = node_weights[node_inputs[0]]

        if node_inputs[1] in tensor_grap:
            self.second_operand = tensor_grap[node_inputs[1]]
            new_shape = [0, self.second_operand.shape.ndims-1] + [i for i in range(1, self.second_operand.shape.ndims-1)]
            self.second_operand = tf.transpose(self.second_operand, perm=new_shape)
        else:
            self.second_operand = node_weights[node_inputs[1]]

    def __call__(self, *args, **kwargs):
        out = tf.matmul(self.first_operand, self.second_operand)
        out = dimension_utils.tensor_NCD_to_NDC_format(out)
        return out

@OPERATOR.register_operator("Pow")
class TFPow():
    def __init__(self, tensor_grap, node_weights, node_inputs, *args, **kwargs):
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

@OPERATOR.register_operator("ReduceSum")
class TFReduceSum():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.keep_dims = node_attribute.get("keepdims", 1) == 1
        input_shape_len = len(tensor_grap[node_inputs[0]].shape)
        self.axes = [dimension_utils.channel_to_last_dimension(i) if i >=0 else dimension_utils.channel_to_last_dimension(input_shape_len + i) for i in node_attribute.get("axes", [-1])]

    def __call__(self, inputs, *args, **kwargs):
        return tf.math.reduce_sum(inputs, axis=self.axes, keepdims=self.keep_dims)

@OPERATOR.register_operator("ReduceMean")
class TFReduceMean():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.keep_dims = node_attribute.get("keepdims", 1) == 1
        input_shape_len = len(tensor_grap[node_inputs[0]].shape)
        self.axes = [dimension_utils.channel_to_last_dimension(i) if i >=0 else dimension_utils.channel_to_last_dimension(input_shape_len + i) for i in node_attribute.get("axes", [-1])]

    def __call__(self, inputs, *args, **kwargs):
        return tf.math.reduce_mean(inputs, axis=self.axes, keepdims=self.keep_dims)

@OPERATOR.register_operator("ReduceMax")
class TFReduceMax():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.keep_dims = node_attribute.get("keepdims", 1) == 1
        input_shape_len = len(tensor_grap[node_inputs[0]].shape)
        self.axes = [dimension_utils.channel_to_last_dimension(i) if i >=0 else dimension_utils.channel_to_last_dimension(input_shape_len + i) for i in node_attribute.get("axes", [-1])]

    def __call__(self, inputs, *args, **kwargs):
        return tf.math.reduce_max(inputs, axis=self.axes, keepdims=self.keep_dims)

@OPERATOR.register_operator("ReduceMin")
class TFReduceMin():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.keep_dims = node_attribute.get("keepdims", 1) == 1
        input_shape_len = len(tensor_grap[node_inputs[0]].shape)
        self.axes = [dimension_utils.channel_to_last_dimension(i) if i >=0 else input_shape_len + i for i in node_attribute.get("axes", [-1])]

    def __call__(self, inputs, *args, **kwargs):
        return tf.math.reduce_min(inputs, axis=self.axes, keepdims=self.keep_dims)

@OPERATOR.register_operator("ArgMax")
class TFArgMax():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.axis = dimension_utils.channel_to_last_dimension(node_attribute.get('axis', 0))
        self.keepdims = node_attribute.get("keepdims", 1) == 1

    def __call__(self, inputs, *args, **kwargs):
        _inputs = tf.argmax(inputs, axis=self.axis)
        if self.keepdims:
            _inputs = tf.expand_dims(_inputs, axis=self.axis)
        return _inputs

@OPERATOR.register_operator("ArgMin")
class TFArgMin():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.axis = dimension_utils.channel_to_last_dimension(node_attribute.get('axis', 0))
        self.keepdims = node_attribute.get("keepdims", 1) == 1

    def __call__(self, inputs, *args, **kwargs):
        _inputs = tf.argmax(inputs, axis=self.axis)
        if self.keepdims:
            _inputs = tf.expand_dims(_inputs, axis=self.axis)
        return _inputs

@OPERATOR.register_operator("Erf")
class TFErf():
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def __call__(self, inputs):
        inputs = tf.math.erf(inputs)
        return inputs