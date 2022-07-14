import logging
import numpy as np
import tensorflow as tf

from . import OPERATOR
from . import shape_axis_utils

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
            second_operand = shape_axis_utils.TorchWeights2TF(second_operand)
    elif (not first_operand_flg) and second_operand_flg:
        # 当second_operand为计算得出的，first_operand来自weight时
        if len(first_operand.shape) == 1:
            # shape=1时，在torch和numpy中因为channel在最后，因此可以利用广播机制进行计算
            first_operand = first_operand[np.newaxis, ...]
            for _ in range(len(second_operand.shape) - 2):
                first_operand = first_operand[..., np.newaxis]
        else:
            first_operand = shape_axis_utils.TorchWeights2TF(first_operand)

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
        self.axes = [shape_axis_utils.Torch2TFAxis(i) if i >=0 else shape_axis_utils.Torch2TFAxis(input_shape_len + i) for i in node_attribute.get("axes", [-1])]

    def __call__(self, inputs, *args, **kwargs):
        return tf.math.reduce_mean(inputs, axis=self.axes, keepdims=self.keep_dims)

@OPERATOR.register_operator("ReduceMax")
class TFReduceMax():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.keep_dims = node_attribute.get("keepdims", 1) == 1
        input_shape_len = len(tensor_grap[node_inputs[0]].shape)
        self.axes = [shape_axis_utils.Torch2TFAxis(i) if i >=0 else shape_axis_utils.Torch2TFAxis(input_shape_len + i) for i in node_attribute.get("axes", [-1])]

    def __call__(self, inputs, *args, **kwargs):
        return tf.math.reduce_max(inputs, axis=self.axes, keepdims=self.keep_dims)

@OPERATOR.register_operator("ReduceMin")
class TFReduceMin():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.keep_dims = node_attribute.get("keepdims", 1) == 1
        input_shape_len = len(tensor_grap[node_inputs[0]].shape)
        self.axes = [shape_axis_utils.Torch2TFAxis(i) if i >=0 else input_shape_len + i for i in node_attribute.get("axes", [-1])]

    def __call__(self, inputs, *args, **kwargs):
        return tf.math.reduce_min(inputs, axis=self.axes, keepdims=self.keep_dims)

@OPERATOR.register_operator("ArgMax")
class TFArgMax():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.axis = shape_axis_utils.Torch2TFAxis(node_attribute['axis'])
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
        self.axis = shape_axis_utils.Torch2TFAxis(node_attribute['axis'])
        self.keepdims = node_attribute.get("keepdims", 1) == 1

    def __call__(self, inputs, *args, **kwargs):
        _inputs = tf.argmax(inputs, axis=self.axis)
        if self.keepdims:
            _inputs = tf.expand_dims(_inputs, axis=self.axis)
        return _inputs

class TFErf():
    def __init__(self) -> None:
        '''
        被注释的部分为完全体,精度高,但计算速度较慢，
        可以根据需求进行删减
        '''
        # self.coefficient = [-1.3026537197817094, 6.4196979235649026e-1,
        #             1.9476473204185836e-2,-9.561514786808631e-3,-9.46595344482036e-4,
        #             3.66839497852761e-4,4.2523324806907e-5,-2.0278578112534e-5,
        #             -1.624290004647e-6,1.303655835580e-6,1.5626441722e-8,-8.5238095915e-8,
        #             6.529054439e-9,5.059343495e-9,-9.91364156e-10,-2.27365122e-10,
        #             9.6467911e-11, 2.394038e-12,-6.886027e-12,8.94487e-13, 3.13092e-13,
        #             -1.12708e-13,3.81e-16,7.106e-15,-1.523e-15,-9.4e-17,1.21e-16,-2.8e-17]
        self.coefficient = [-1.3026537197817094, 6.4196979235649026e-1,
                    1.9476473204185836e-2,-9.561514786808631e-3,-9.46595344482036e-4,
                    3.66839497852761e-4,4.2523324806907e-5,-2.0278578112534e-5]
    
    def __call__(self, inputs):
        inputs = tf.where(inputs>=0, 1-self.erfccheb(inputs), self.erfccheb(-inputs)-1)
        return inputs

    def erfccheb(self, data):
        data = tf.maximum(data, 0)
        var1 = 2.0/(2.0 + data)
        var2 = 4.0*var1 - 2.0
        var3, var4, var5 = 0.0, 0.0, 0.0
        for j in range(len(self.coefficient)-1, 0, -1):
            var3 = var4
            var4 = var2*var4 - var5 + self.coefficient[j]
            var5 = var3
        return var1*tf.exp(-data*data + 0.5*(self.coefficient[0] + var2*var4) - var5)