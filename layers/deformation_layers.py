import logging
import tensorflow as tf

from . import OPERATOR
from . import dimension_utils

LOG = logging.getLogger("deformation_layers :")

@OPERATOR.register_operator("Transpose")
class TFTranspose():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.trans_in, self.trans_out = None, None
        if kwargs.get("perm_list"):
            self.perm_list = kwargs.get("perm_list")
        elif len(node_attribute['perm']) > 4:
            self.perm_list = []
            for axis in node_attribute['perm']:
                new_axis = dimension_utils.channel_to_last_dimension(axis)
                if new_axis == -1:
                    new_axis = max(node_attribute['perm'])
                self.perm_list.append(new_axis)
            self.perm_list = dimension_utils.shape_NCD_to_NDC_format(self.perm_list)
        else:
            self.perm_list = [i for i in node_attribute['perm']]
            LOG.info("Transpose will process tensor after change back to NCHW format.")
            shape_len = len(tensor_grap[node_inputs[0]].shape)
            self.trans_in = [0, shape_len-1] + [n for n in range(1, shape_len-1)]
            self.trans_out = [0] + [n for n in range(2, len(self.perm_list))] + [1]

    def __call__(self, inputs):
        if self.trans_in and self.trans_out:
            inputs = tf.transpose(inputs, perm=self.trans_in)
            inputs = tf.transpose(inputs, perm=self.perm_list)
            inputs = tf.transpose(inputs, perm=self.trans_out)
            return inputs
        else:
            return tf.transpose(inputs, perm=self.perm_list)

@OPERATOR.register_operator("Slice")
class TFSlice():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        if len(node_inputs) == 1:
            self.starts = node_attribute['starts'][0]
            self.ends = node_attribute['ends'][0]
            self.axis = dimension_utils.channel_to_last_dimension(node_attribute['axes'][0])
            self.steps = 1
        else:
            self.starts = node_weights[node_inputs[1]][0] if node_inputs[1] in node_weights else tensor_grap[node_inputs[1]][0]
            self.axis = node_weights[node_inputs[3]][0] if node_inputs[3] in node_weights else tensor_grap[node_inputs[3]][0]
            self.axis = dimension_utils.channel_to_last_dimension(self.axis)
            self.ends = node_weights[node_inputs[2]][0] if node_inputs[2] in node_weights else tensor_grap[node_inputs[2]][0]
            self.ends = min(self.ends, tensor_grap[node_inputs[0]].shape[self.axis])
            if len(node_inputs) < 5:
                self.steps = 1
            else:
                self.steps = node_weights[node_inputs[4]][0] if node_inputs[4] in node_weights else tensor_grap[node_inputs[4]][0]

    def __call__(self, inputs):
        indices = tf.keras.backend.arange(self.starts, self.ends, step=self.steps)
        return tf.gather(inputs, indices, axis=self.axis)

@OPERATOR.register_operator("Gather")
class TFGather():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        self.axis = dimension_utils.channel_to_last_dimension(node_attribute.get('axis', 0))
        self.indices = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]

    def __call__(self, inputs):
        return tf.gather(inputs, self.indices, axis=self.axis)

@OPERATOR.register_operator("Concat")
class TFConcat():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        _axis = dimension_utils.channel_to_last_dimension(node_attribute['axis'])
        _gather = [tensor_grap[x] for x in node_inputs]
        self.out = tf.concat(_gather, axis=_axis)

    def __call__(self, *args, **kwargs):
        return self.out

@OPERATOR.register_operator("Reshape")
class TFReshape():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.out_shape = node_weights[node_inputs[1]]
        self.trans_in, self.trans_out = None, None
        LOG.info("Reshape will process tensor after change back to NCHW format.")
        shape_len = len(tensor_grap[node_inputs[0]].shape)
        self.trans_in = [0, shape_len-1] + [n for n in range(1, shape_len-1)]
        self.trans_out = [0] + [n for n in range(2, len(self.out_shape))] + [1]

    def __call__(self, inputs):
        inputs = tf.transpose(inputs, perm=self.trans_in)
        inputs = tf.reshape(inputs, shape=self.out_shape)
        inputs = tf.transpose(inputs, perm=self.trans_out)
        return inputs
        
@OPERATOR.register_operator("Flatten")
class TFFlatten():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        tensor_size, tensor_shape = 1, tensor_grap[node_inputs[0]].get_shape().as_list()
        for n in tensor_shape:
            tensor_size = tensor_size * max(n, 1)
        if tensor_size == max(tensor_shape):
            self.trans = None
        else:
            perm_list = [0, len(tensor_shape)-1]
            for i in range(len(tensor_shape)-2):
                perm_list.append(i+1)
            self.trans = TFTranspose(None, None, None, None, perm_list=perm_list)

    def __call__(self, inputs):
        if self.trans:
            inputs = self.trans(inputs)
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