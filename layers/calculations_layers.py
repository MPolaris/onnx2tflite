import logging
import numpy as np
import tensorflow as tf

from . import OPERATOR
from . import dimension_utils

LOG = logging.getLogger("calculations_layers :")

def get_number(tensor_grap, node_weights, node_inputs):
    first_operand, second_operand = None, None
    # 标识opreand来自weight还是前面node计算得出的
    first_operand_flg, second_operand_flg = True, True
    if node_inputs[0] in tensor_grap:
        first_operand = tensor_grap[node_inputs[0]]
    else:
        first_operand = node_weights[node_inputs[0]]
        first_operand_flg = False

    if node_inputs[1] in tensor_grap:
        second_operand = tensor_grap[node_inputs[1]]
    else:
        second_operand = node_weights[node_inputs[1]]
        second_operand_flg = False

    if first_operand_flg and (not second_operand_flg):
        # 当first_operand为计算得出的，second_operand来自weight时
        if len(second_operand.shape) == 1:
            # shape=1时，在torch和numpy中因为channel在最后，因此可以利用广播机制进行计算
            second_operand = second_operand[np.newaxis, ...]
            for _ in range(len(first_operand.shape) - 2):
                second_operand = second_operand[..., np.newaxis]
        else:
            second_operand = dimension_utils.tensor_NCD_to_NDC_format(second_operand)
    elif (not first_operand_flg) and second_operand_flg:
        # 当second_operand为计算得出的，first_operand来自weight时
        if len(first_operand.shape) == 1:
            # shape=1时，在torch和numpy中因为channel在最后，因此可以利用广播机制进行计算
            first_operand = first_operand[np.newaxis, ...]
            for _ in range(len(second_operand.shape) - 2):
                first_operand = first_operand[..., np.newaxis]
        else:
            first_operand = dimension_utils.tensor_NCD_to_NDC_format(first_operand)

    return first_operand, second_operand

@OPERATOR.register_operator("Add")
class TFAdd():
    def __init__(self, tensor_grap, node_weights, node_inputs, *args, **kwargs):
        super().__init__()
        self.first_operand, self.second_operand = get_number(tensor_grap, node_weights, node_inputs)

    def __call__(self, *args, **kwargs):
        return self.first_operand + self.second_operand

@OPERATOR.register_operator("Sub")
class TFSub():
    def __init__(self, tensor_grap, node_weights, node_inputs, *args, **kwargs):
        super().__init__()
        self.first_operand, self.second_operand = get_number(tensor_grap, node_weights, node_inputs)

    def __call__(self, *args, **kwargs):
        return self.first_operand - self.second_operand

@OPERATOR.register_operator("Mul")
class TFMul():
    def __init__(self, tensor_grap, node_weights, node_inputs, *args, **kwargs):
        super().__init__()
        self.first_operand, self.second_operand = get_number(tensor_grap, node_weights, node_inputs)

    def __call__(self, *args, **kwargs):
        return self.first_operand * self.second_operand

@OPERATOR.register_operator("Div")
class TFDiv():
    def __init__(self, tensor_grap, node_weights, node_inputs, *args, **kwargs):
        super().__init__()
        self.first_operand, self.second_operand = get_number(tensor_grap, node_weights, node_inputs)

    def __call__(self, *args, **kwargs):
        return self.first_operand / self.second_operand

@OPERATOR.register_operator("MatMul")
class TFMatMul():
    def __init__(self, tensor_grap, node_weights, node_inputs, *args, **kwargs):
        super().__init__()
        self.first_operand, self.second_operand = get_number(tensor_grap, node_weights, node_inputs)
        self.trans_in, self.trans_out = None, None
        if self.first_operand.shape[-1] != self.second_operand.shape[0] or len(self.second_operand.shape) == 2:
            # channel轴变化遇上广播，这感觉真的来劲
            shape_len = len(tensor_grap[node_inputs[0]].shape)
            self.trans_in = [0, shape_len-1] + [n for n in range(1, shape_len-1)]
            self.trans_out = [0] + [n for n in range(2, len(self.first_operand.shape))] + [1]

    def __call__(self, *args, **kwargs):
        if self.trans_in:
            temp = tf.transpose(self.first_operand, perm=self.trans_in)
            temp = tf.matmul(temp, self.second_operand)
            temp = tf.transpose(temp, perm=self.trans_out)
            return temp
        else:
            return tf.matmul(self.first_operand, self.second_operand)

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