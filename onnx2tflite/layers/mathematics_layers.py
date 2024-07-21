import logging
import numpy as np
import tensorflow as tf

from onnx2tflite.utils.definitions import Layout
from onnx2tflite.utils import OPERATOR, dimension_utils, np2tf_type

LOG = logging.getLogger("calculations_layers :")

def np2tf(x):
    if isinstance(x, np.ndarray):
        x = tf.convert_to_tensor(x, dtype=np2tf_type[x.dtype.name])
        return x, False
    return x, True

def match_tensor(x1:tf.Tensor or np.ndarray, x2:tf.Tensor or np.ndarray, x1_layout:Layout, x2_layout:Layout):
    
    x1, f1 = np2tf(x1)
    x2, f2 = np2tf(x2)

    # no need to transpose if all var are tensor, we assume tensor are computed by gragh.
    if f1 and f2:
        if x1_layout != x2_layout:
            if x1_layout == Layout.Channel_Last:
                x1 = dimension_utils.tensor_NDC_to_NCD_format(x1)
            elif x2_layout == Layout.Channel_Last:
                x2 = dimension_utils.tensor_NDC_to_NCD_format(x2)
        return x1, x2, Layout.Channel_First
    
    # ensure tensor is set to x1, const weights set to x2
    out_layout = x1_layout
    if f2:
        x1, x2 = x2, x1
        out_layout = x2_layout 
    

    if out_layout == Layout.Channel_Last:
        if x1.shape.ndims != x2.shape.ndims:
            while x2.shape.ndims < x1.shape.ndims:
                x2 = tf.expand_dims(x2, axis=0)
        x2 = dimension_utils.tensor_NCD_to_NDC_format(x2)
        
    x2 = tf.cast(x2, x1.dtype)
    return (x2, x1, out_layout) if f2 else (x1, x2, out_layout)

'''
tensor(NDC) + const
tensor(NCD) + const
tensor(NDC) + tensor(NDC)
tensor(NCD) + tensor(NCD)
'''

class BaseArithmetic:
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs):
        self.left_val, self.right_val = None, None
        left_layout, right_layout = Layout.Default, Layout.Default

        if node_inputs[0] in tensor_grap:
            self.left_val = tensor_grap[node_inputs[0]]
            left_layout = layout_dict[node_inputs[0]]
        else:
            self.left_val = node_weights[node_inputs[0]]

        if node_inputs[1] in tensor_grap:
            self.right_val = tensor_grap[node_inputs[1]]
            right_layout = layout_dict[node_inputs[1]]
        else:
            self.right_val = node_weights[node_inputs[1]]
        
        if left_layout == right_layout:
            return
        
        self.left_val, self.right_val, out_layout = match_tensor(self.left_val, self.right_val, left_layout, right_layout)
        layout_dict[node_outputs[0]] = out_layout

@OPERATOR.register_operator("Add")
class TFAdd(BaseArithmetic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.left_val + self.right_val

@OPERATOR.register_operator("Sub")
class TFSub(BaseArithmetic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.left_val - self.right_val

@OPERATOR.register_operator("Mul")
class TFMul(BaseArithmetic):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.left_val * self.right_val

@OPERATOR.register_operator("Div")
class TFDiv(BaseArithmetic):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.left_val / self.right_val

@OPERATOR.register_operator("MatMul")
class TFMatMul():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs):
        super().__init__()
        if node_inputs[0] in tensor_grap:
            self.A = tensor_grap[node_inputs[0]]
            if layout_dict[node_inputs[0]] == Layout.Channel_Last:
                self.A = dimension_utils.tensor_NDC_to_NCD_format(self.A)
        else:
            self.A = node_weights[node_inputs[0]]

        if node_inputs[1] in tensor_grap:
            self.B = tensor_grap[node_inputs[1]]
            if layout_dict[node_inputs[1]] == Layout.Channel_Last:
                self.B = dimension_utils.tensor_NDC_to_NCD_format(self.B)
        else:
            self.B = node_weights[node_inputs[1]]

        self.dense = tf.keras.layers.Dense(self.B.shape[-1],
                                            weights=[self.B],
                                            use_bias=False)

        layout_dict[node_outputs[0]] = Layout.Channel_First

    def __call__(self, *args, **kwargs):
        # out = tf.matmul(self.A, self.B)
        try:
            out = self.dense(self.A)
        except Exception:
            out = tf.matmul(self.A, self.B)
        return out

@OPERATOR.register_operator("Mod")
class TFMod():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.fmod = bool(node_attribute.get("fmod", 0))
        self.mod_value = None
        if node_inputs[1] in node_weights:
            self.mod_value = node_weights[node_inputs[1]]
        else:
            self.mod_value = tensor_grap[node_inputs[1]]

    def __call__(self, inputs):
        if self.fmod:
            return tf.math.floormod(inputs, tf.cast(self.mod_value, inputs.dtype))
        else:
            return tf.math.mod(inputs, tf.cast(self.mod_value, inputs.dtype))

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

class ReduceBase:
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs):
        self.keep_dims = node_attribute.get("keepdims", 1) == 1
        input_shape_len = len(tensor_grap[node_inputs[0]].shape)
        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.axes = [dimension_utils.channel_to_last_dimension(i) if i >=0 else dimension_utils.channel_to_last_dimension(input_shape_len + i) for i in node_attribute.get("axes", [-1])]
        else:
            self.axes = [i if i >=0 else input_shape_len + i for i in node_attribute.get("axes", [-1])]

@OPERATOR.register_operator("ReduceSum")
class TFReduceSum(ReduceBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, inputs, *args, **kwargs):
        return tf.math.reduce_sum(inputs, axis=self.axes, keepdims=self.keep_dims)

@OPERATOR.register_operator("ReduceMean")
class TFReduceMean(ReduceBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, inputs, *args, **kwargs):
        return tf.math.reduce_mean(inputs, axis=self.axes, keepdims=self.keep_dims)

@OPERATOR.register_operator("ReduceMax")
class TFReduceMax(ReduceBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, inputs, *args, **kwargs):
        return tf.math.reduce_max(inputs, axis=self.axes, keepdims=self.keep_dims)

@OPERATOR.register_operator("ReduceMin")
class TFReduceMin(ReduceBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, inputs, *args, **kwargs):
        return tf.math.reduce_min(inputs, axis=self.axes, keepdims=self.keep_dims)

@OPERATOR.register_operator("ArgMax")
class TFArgMax():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs):
        super().__init__()
        self.axis = node_attribute.get('axis', 0)
        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.axis = dimension_utils.channel_to_last_dimension(self.axis)
        self.keepdims = node_attribute.get("keepdims", 1) == 1

    def __call__(self, inputs, *args, **kwargs):
        _inputs = tf.argmax(inputs, axis=self.axis)
        if self.keepdims:
            _inputs = tf.expand_dims(_inputs, axis=self.axis)
        return _inputs

@OPERATOR.register_operator("ArgMin")
class TFArgMin():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs):
        super().__init__()
        self.axis = node_attribute.get('axis', 0)
        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.axis = dimension_utils.channel_to_last_dimension(self.axis)
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