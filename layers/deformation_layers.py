import logging
import tensorflow as tf

from . import OPERATOR
from . import dimension_utils

LOG = logging.getLogger("deformation_layers :")

@OPERATOR.register_operator("Transpose")
class TFTranspose():
    def __init__(self, tensor_grap, node_inputs, node_attribute, node_outputs, runtime_format, *args, **kwargs)->None:
        super().__init__()
        self.transpose = None
        if runtime_format.get(node_inputs[0]) != "ONNX":
            self.transpose = dimension_utils.tensor_NDC_to_NCD_format
        for no in node_outputs:
            runtime_format[no] = "ONNX"
        
        self.perm_list = [i for i in node_attribute['perm']]

    def __call__(self, inputs):
        if self.transpose:
            inputs = tf.transpose(inputs, perm=self.trans_in)
        inputs = tf.transpose(inputs, perm=self.perm_list)
        return inputs

@OPERATOR.register_operator("Slice")
class TFSlice():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, runtime_format, *args, **kwargs) -> None:
        super().__init__()
        if len(node_inputs) == 1:
            self.starts = node_attribute['starts'][0]
            self.ends = node_attribute['ends'][0]
            self.axis = node_attribute['axes'][0]
            if runtime_format[node_inputs[0]] != "ONNX":
                self.axis = dimension_utils.channel_to_last_dimension(self.axis)
            self.steps = 1
        else:
            self.starts = node_weights[node_inputs[1]][0] if node_inputs[1] in node_weights else tensor_grap[node_inputs[1]][0]
            self.axis = node_weights[node_inputs[3]][0] if node_inputs[3] in node_weights else tensor_grap[node_inputs[3]][0]
            if runtime_format[node_inputs[0]] != "ONNX":
                self.axis = dimension_utils.channel_to_last_dimension(self.axis)
            self.ends = node_weights[node_inputs[2]][0] if node_inputs[2] in node_weights else tensor_grap[node_inputs[2]][0]
            self.ends = min(self.ends, tensor_grap[node_inputs[0]].shape[self.axis])
            if self.ends < 0:
                self.ends = tensor_grap[node_inputs[0]].shape[self.axis] + self.ends
            if len(node_inputs) < 5:
                self.steps = 1
            else:
                self.steps = node_weights[node_inputs[4]][0] if node_inputs[4] in node_weights else tensor_grap[node_inputs[4]][0]
        
        runtime_format[node_outputs[0]] = runtime_format[node_inputs[0]]

    def __call__(self, inputs):
        indices = tf.keras.backend.arange(self.starts, self.ends, step=self.steps)
        res = tf.gather(inputs, indices, axis=self.axis)
        return res

@OPERATOR.register_operator("Gather")
class TFGather():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, runtime_format, *args, **kwargs) -> None:
        super().__init__()
        self.axis = node_attribute.get('axis', 0)
        if runtime_format[node_inputs[0]] != "ONNX":
            self.axis = dimension_utils.channel_to_last_dimension(self.axis)

        runtime_format[node_outputs[0]] = runtime_format[node_inputs[0]]

        self.indices = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]

    def __call__(self, inputs):
        return tf.gather(inputs, self.indices, axis=self.axis)

@OPERATOR.register_operator("Concat")
class TFConcat():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, runtime_format, *args, **kwargs):
        super().__init__()
        self._axis = node_attribute['axis']
        if runtime_format[node_inputs[0]] != "ONNX":
            self._axis = dimension_utils.channel_to_last_dimension(self._axis)
        self._gather = [tensor_grap[x] if x in tensor_grap else node_weights[x] for x in node_inputs]
        runtime_format[node_outputs[0]] = runtime_format[node_inputs[0]]

    def __call__(self, *args, **kwargs):
        return tf.concat(self._gather, axis=self._axis)

@OPERATOR.register_operator("Reshape")
class TFReshape():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, runtime_format, *args, **kwargs):
        super().__init__()
        self.transpose = None
        if runtime_format.get(node_inputs[0]) != "ONNX":
            self.transpose = dimension_utils.tensor_NDC_to_NCD_format
        for no in node_outputs:
            runtime_format[no] = "ONNX"

        self.out_shape = node_weights[node_inputs[1]]

    def __call__(self, inputs):
        if self.transpose:
            inputs = self.transpose(inputs)
        inputs = tf.reshape(inputs, shape=self.out_shape)
        return inputs
        
@OPERATOR.register_operator("Flatten")
class TFFlatten():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        tensor_size, tensor_shape = 1, tensor_grap[node_inputs[0]].get_shape().as_list()
        for n in tensor_shape:
            tensor_size = tensor_size * max(n, 1)
        if tensor_size == max(tensor_shape):
            self.perm_list = None
        else:
            perm_list = [0, len(tensor_shape)-1]
            for i in range(len(tensor_shape)-2):
                perm_list.append(i+1)
            self.perm_list = perm_list

    def __call__(self, inputs):
        if self.perm_list:
            inputs = tf.transpose(inputs, perm=self.perm_list)
        return tf.reshape(inputs, shape=(1, -1))

@OPERATOR.register_operator("Split")
class TFSplit():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        index = kwargs.get('index', 0)
        start = 0
        for i in range(index):
            start += int(node_attribute['split'][i])
        end = start + node_attribute['split'][index]
        self.indices = tf.keras.backend.arange(start, end, 1)
        self.axis = dimension_utils.channel_to_last_dimension(node_attribute.get("axis", 0))

    def __call__(self, inputs):
        return tf.gather(inputs, indices=self.indices, axis=self.axis)

@OPERATOR.register_operator("Expand")
class TFExpand():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.shape = dimension_utils.shape_NCD_to_NDC_format(node_weights[node_inputs[1]])

    def __call__(self, inputs):
        for i in range(len(self.shape)):
            if int(self.shape[i]//inputs.shape[i]) > 1:
                inputs = tf.repeat(inputs, repeats=int(self.shape[i]//inputs.shape[i]), axis=i)
            elif self.shape[i] < inputs.shape[i] and self.shape[i] != 1:
                inputs = tf.repeat(inputs, repeats=int(self.shape[i]), axis=i)
        return inputs

@OPERATOR.register_operator("Unsqueeze")
class TFUnsqueeze():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.axis = dimension_utils.channel_to_last_dimension(node_attribute['axes'][0])

    def __call__(self, inputs):
        return tf.expand_dims(inputs, self.axis)

@OPERATOR.register_operator("Squeeze")
class TFSqueeze():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.axis = dimension_utils.channel_to_last_dimension(node_attribute['axes'][0])

    def __call__(self, inputs):
        return tf.squeeze(inputs, self.axis)