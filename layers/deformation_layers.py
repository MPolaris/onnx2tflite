import numpy as np
import tensorflow as tf
import logging
from tensorflow import keras

from . import OPERATOR
from . import shape_axis_utils

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
                new_axis = shape_axis_utils.Torch2TFAxis(axis)
                if new_axis == -1:
                    new_axis = max(node_attribute['perm'])
                self.perm_list.append(new_axis)
            self.perm_list = shape_axis_utils.TorchShape2TF(self.perm_list)
        else:
            self.perm_list = [i for i in node_attribute['perm']]
            LOG.warning("Transpose 操作将会回到NCHW形式进行")
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
            self.axis = shape_axis_utils.Torch2TFAxis(node_attribute['axes'][0])
            self.steps = 1
        else:
            self.starts = node_weights[node_inputs[1]][0]
            self.axis = shape_axis_utils.Torch2TFAxis(node_weights[node_inputs[3]][0])
            self.ends = min(node_weights[node_inputs[2]][0], tensor_grap[node_inputs[0]].shape[self.axis])
            self.steps = 1 if len(node_inputs) < 5 else node_weights[node_inputs[4]][0]

    def __call__(self, inputs):
        indices = tf.keras.backend.arange(self.starts, self.ends, step=self.steps)
        return tf.gather(inputs, indices, axis=self.axis)

@OPERATOR.register_operator("Gather")
class TFGather():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        self.axis = shape_axis_utils.Torch2TFAxis(node_attribute['axis'])
        self.indices = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]

    def __call__(self, inputs):
        return tf.gather(inputs, self.indices, axis=self.axis)

@OPERATOR.register_operator("Concat")
class TFConcat():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        _axis = shape_axis_utils.Torch2TFAxis(node_attribute['axis'])
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
        LOG.warning("Reshape 操作将会回到NCHW形式进行")
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