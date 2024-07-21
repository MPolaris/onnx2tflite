import numpy as np
import tensorflow as tf
from tensorflow import keras

from onnx2tflite.utils.definitions import Layout
from onnx2tflite.utils import OPERATOR, channel_to_last_dimension, tensor_NCD_to_NDC_format

@OPERATOR.register_operator("Relu")
class TFRelu():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return keras.activations.relu(inputs)

@OPERATOR.register_operator("HardSigmoid")
class TFHardSigmoid():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        self.alpha = node_attribute.get("alpha", 0.2)
        self.beta = node_attribute.get("beta", 0.5)

    def __call__(self, inputs):
        return tf.clip_by_value(self.alpha*inputs+self.beta, 0, 1)

@OPERATOR.register_operator("HardSwish")
class TFHardSwish():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return inputs*tf.clip_by_value(inputs/6+0.5, 0, 1)

@OPERATOR.register_operator("Mish")
class TFMish():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return inputs*tf.tanh(tf.math.log(tf.math.exp(inputs)+1))

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
        self.alpha = node_attribute.get('alpha', 0.01)

    def __call__(self, inputs):
        return keras.activations.relu(inputs, alpha=self.alpha)

@OPERATOR.register_operator("PRelu")
class TFPRelu():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs) -> None:
        super().__init__()
        if 'slope' in node_attribute:
            self.slope = node_attribute['slope']
        elif node_inputs[1] in node_weights:
            self.slope = node_weights[node_inputs[1]]
        else:
            self.slope = tensor_grap[node_inputs[1]]
        input_tensor_shape = tensor_grap[node_inputs[0]].shape
        channel_last = layout_dict[node_inputs[0]] == Layout.Channel_Last
        if isinstance(self.slope, np.ndarray):
            while self.slope.ndim < input_tensor_shape.ndims:
                self.slope = self.slope[np.newaxis, :]
            if channel_last:
                self.slope = tensor_NCD_to_NDC_format(self.slope)
            if self.slope.ndim > 1:
                # remove batchsize
                self.slope = self.slope[0]
        axes = [i for i in range(1, input_tensor_shape.ndims-1)] if channel_last else [i for i in range(2, input_tensor_shape.ndims)]
        self.PRelu = tf.keras.layers.PReLU(weights=[self.slope], shared_axes = axes)

    def __call__(self, inputs):
        return self.PRelu(inputs)

@OPERATOR.register_operator("Sin")
class TFSin():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return tf.sin(inputs)

@OPERATOR.register_operator("Sinh")
class TFSinh():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return tf.sinh(inputs)

@OPERATOR.register_operator("Cos")
class TFCos():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return tf.cos(inputs)

@OPERATOR.register_operator("Cosh")
class TFCosh():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return tf.cosh(inputs)

@OPERATOR.register_operator("Tan")
class TFTan():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return tf.tan(inputs)

@OPERATOR.register_operator("Tanh")
class TFTanh():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return tf.tanh(inputs)

@OPERATOR.register_operator("Softmax")
class TFSoftmax():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs) -> None:
        super().__init__()
        self.axis = node_attribute.get('axis', -1)
        if self.axis == -1:
            self.axis = len(tensor_grap[node_inputs[0]].shape.as_list()) - 1
        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.axis = channel_to_last_dimension(self.axis)

    def __call__(self, inputs):
        return keras.activations.softmax(inputs, axis=self.axis)

@OPERATOR.register_operator("Softplus")
class TFSoftplus():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return keras.activations.softplus(inputs)
    
@OPERATOR.register_operator("Softsign")
class TFSoftsign():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return keras.activations.softsign(inputs)

@OPERATOR.register_operator("Selu")
class TFSelu():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return keras.activations.selu(inputs)
    
@OPERATOR.register_operator("Elu")
class TFElu():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return keras.activations.elu(inputs)
    
@OPERATOR.register_operator("Celu")
class TFCelu():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        self.alpha = node_attribute.get("alpha", 1.0)

    def __call__(self, inputs):
        return tf.maximum(inputs, 0) + tf.minimum(0, self.alpha*(tf.exp(inputs/self.alpha)-1))