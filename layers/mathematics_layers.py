import logging
import numpy as np
import tensorflow as tf

from . import OPERATOR
from . import dimension_utils

LOG = logging.getLogger("calculations_layers :")

def get_number(tensor_grap, node_weights, node_inputs, node_outputs, runtime_format):
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

    if isinstance(first_operand, (int, float)) or isinstance(second_operand, (int, float)) or \
                    runtime_format[node_inputs[0]] == runtime_format[node_inputs[1]]:
        runtime_format[node_outputs[0]] = runtime_format[node_inputs[0]] if first_operand_flg else runtime_format[node_inputs[1]]
        return first_operand, second_operand
    
    if first_operand.ndim == second_operand.ndim:
        if first_operand_flg == second_operand_flg:
            runtime_format[node_outputs[0]] = runtime_format[node_inputs[0]] if first_operand_flg else runtime_format[node_inputs[1]]
            return first_operand, second_operand
        
        if first_operand_flg and runtime_format[node_inputs[0]] != "ONNX":
            second_operand = dimension_utils.tensor_NCD_to_NDC_format(second_operand)
            runtime_format[node_outputs[0]] = "TFLITE"
            return first_operand, second_operand
        
        if second_operand_flg and runtime_format[node_inputs[1]] != "ONNX":
            first_operand = dimension_utils.tensor_NCD_to_NDC_format(first_operand)
            runtime_format[node_outputs[0]] = "TFLITE"
            return first_operand, second_operand
    else:
        if runtime_format[node_inputs[0]] != "ONNX":
            first_operand = dimension_utils.tensor_NDC_to_NCD_format(first_operand)

        if runtime_format[node_inputs[1]] != "ONNX":
            second_operand = dimension_utils.tensor_NDC_to_NCD_format(second_operand)

        runtime_format[node_outputs[0]] = "ONNX"
        return first_operand, second_operand

@OPERATOR.register_operator("Add")
class TFAdd():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_outputs, runtime_format, *args, **kwargs):
        super().__init__()
        self.first_operand, self.second_operand = get_number(tensor_grap, node_weights, node_inputs, node_outputs, runtime_format)

    def __call__(self, *args, **kwargs):
        return self.first_operand + self.second_operand

@OPERATOR.register_operator("Sub")
class TFSub():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_outputs, runtime_format, *args, **kwargs):
        super().__init__()
        self.first_operand, self.second_operand = get_number(tensor_grap, node_weights, node_inputs, node_outputs, runtime_format)

    def __call__(self, *args, **kwargs):
        return self.first_operand - self.second_operand

@OPERATOR.register_operator("Mul")
class TFMul():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_outputs, runtime_format, *args, **kwargs):
        super().__init__()
        self.first_operand, self.second_operand = get_number(tensor_grap, node_weights, node_inputs, node_outputs, runtime_format)

    def __call__(self, *args, **kwargs):
        return self.first_operand * self.second_operand

@OPERATOR.register_operator("Div")
class TFDiv():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_outputs, runtime_format, *args, **kwargs):
        super().__init__()
        self.first_operand, self.second_operand = get_number(tensor_grap, node_weights, node_inputs, node_outputs, runtime_format)

    def __call__(self, *args, **kwargs):
        return self.first_operand / self.second_operand

@OPERATOR.register_operator("MatMul")
class TFMatMul():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_outputs, runtime_format, *args, **kwargs):
        super().__init__()
        self.first_operand, self.second_operand = get_number(tensor_grap, node_weights, node_inputs, node_outputs, runtime_format)
        self.dense = None
        if isinstance(self.second_operand, np.ndarray):
            self.dense = tf.keras.layers.Dense(
                self.second_operand.shape[-1],
                weights=[self.second_operand], bias_initializer='zeros', kernel_initializer='zeros', use_bias=False
            )
        else:
            print(1)

    def __call__(self, *args, **kwargs):
        if self.dense:
            return self.dense(self.first_operand)
        else:
            return tf.raw_ops.BatchMatMul(x=self.first_operand, y=self.second_operand)

@OPERATOR.register_operator("Pow")
class TFPow():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_outputs, runtime_format, *args, **kwargs):
        super().__init__()
        self.power_index = node_weights[node_inputs[1]]
        runtime_format[node_outputs[0]] = runtime_format[node_inputs[0]]

    def __call__(self, inputs, *args, **kwargs):
        return tf.pow(inputs, self.power_index)

@OPERATOR.register_operator("Reciprocal")
class TFReciprocal():
    def __init__(self, node_inputs, node_outputs, runtime_format, *args, **kwargs):
        super().__init__()
        runtime_format[node_outputs[0]] = runtime_format[node_inputs[0]]

    def __call__(self, inputs, *args, **kwargs):
        return 1/inputs

@OPERATOR.register_operator("Sqrt")
class TFSqrt():
    def __init__(self, node_inputs, node_outputs, runtime_format, *args, **kwargs):
        super().__init__()
        runtime_format[node_outputs[0]] = runtime_format[node_inputs[0]]

    def __call__(self, inputs, *args, **kwargs):
        return tf.sqrt(inputs)

@OPERATOR.register_operator("Exp")
class TFSqrt():
    def __init__(self, node_inputs, node_outputs, runtime_format, *args, **kwargs):
        super().__init__()
        runtime_format[node_outputs[0]] = runtime_format[node_inputs[0]]

    def __call__(self, inputs, *args, **kwargs):
        return tf.exp(inputs)

@OPERATOR.register_operator("Log")
class TFLog():
    def __init__(self, node_inputs, node_outputs, runtime_format, *args, **kwargs):
        super().__init__()
        runtime_format[node_outputs[0]] = runtime_format[node_inputs[0]]

    def __call__(self, inputs, *args, **kwargs):
        return tf.log(inputs)

@OPERATOR.register_operator("ReduceMean")
class TFReduceMean():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, runtime_format, *args, **kwargs):
        super().__init__()
        for no in node_outputs:
            runtime_format[no] = runtime_format[node_inputs[0]]

        self.keep_dims = node_attribute.get("keepdims", 1) == 1
        input_shape_len = len(tensor_grap[node_inputs[0]].shape)

        if runtime_format.get(node_inputs[0]) != "ONNX":
            self.axes = [dimension_utils.channel_to_last_dimension(i) if i >=0 else dimension_utils.channel_to_last_dimension(input_shape_len + i) for i in node_attribute.get("axes", [-1])]
        else:
            self.axes = [i if i >=0 else input_shape_len + i for i in node_attribute.get("axes", [-1])]

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
    def __init__(self, node_inputs, node_outputs, runtime_format, *args, **kwargs) -> None:
        runtime_format[node_outputs[0]] = runtime_format[node_inputs[0]]
    
    def __call__(self, inputs):
        inputs = tf.math.erf(inputs)
        return inputs