import logging
import tensorflow as tf

from onnx2tflite.utils.definitions import Layout
from onnx2tflite.utils import OPERATOR, dimension_utils

LOG = logging.getLogger("deformation_layers :")

@OPERATOR.register_operator("Transpose")
class TFTranspose():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs)->None:
        super().__init__()
        for nop in node_outputs:
            layout_dict[nop] = Layout.Channel_First
        if kwargs.get("perm_list"):
            self.perm_list = kwargs.get("perm_list")
            return
        self.trans_in = None
        self.perm_list = [i for i in node_attribute['perm']]
        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            # LOG.info("Transpose will process tensor after change back to NCHW format.")
            shape_len = len(tensor_grap[node_inputs[0]].shape)
            self.trans_in = [0, shape_len-1] + [n for n in range(1, shape_len-1)]

    def __call__(self, inputs):
        if self.trans_in:
            inputs = tf.transpose(inputs, perm=self.trans_in)
        return tf.transpose(inputs, perm=self.perm_list)

@OPERATOR.register_operator("Slice")
class TFSlice():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs) -> None:
        super().__init__()
        if len(node_inputs) == 1:
            self.starts = node_attribute['starts'][0]
            self.ends = node_attribute['ends'][0]
            self.axis = node_attribute['axes'][0]
            self.steps = 1
        else:
            self.starts = node_weights[node_inputs[1]][0] if node_inputs[1] in node_weights else tensor_grap[node_inputs[1]][0]
            self.axis = node_weights[node_inputs[3]][0] if node_inputs[3] in node_weights else tensor_grap[node_inputs[3]][0]
            self.ends = node_weights[node_inputs[2]][0] if node_inputs[2] in node_weights else tensor_grap[node_inputs[2]][0]
            self.ends = min(self.ends, tensor_grap[node_inputs[0]].shape[self.axis])
            if len(node_inputs) < 5:
                self.steps = 1
            else:
                self.steps = node_weights[node_inputs[4]][0] if node_inputs[4] in node_weights else tensor_grap[node_inputs[4]][0]
        
        shape = tensor_grap[node_inputs[0]].shape.as_list()
        if self.starts < 0:
            self.starts = shape[self.axis] + self.starts
        if self.ends < 0:
            self.ends = shape[self.axis] + self.ends

        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.axis = dimension_utils.channel_to_last_dimension(self.axis)

    def __call__(self, inputs):
        indices = tf.keras.backend.arange(self.starts, self.ends, step=self.steps)
        return tf.gather(inputs, indices, axis=self.axis)

@OPERATOR.register_operator("Gather")
class TFGather():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs) -> None:
        super().__init__()
        self.axis = node_attribute.get('axis', 0)
        self.indices = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.axis = dimension_utils.channel_to_last_dimension(self.axis)

    def __call__(self, inputs):
        return tf.gather(inputs, self.indices, axis=self.axis)

@OPERATOR.register_operator("Concat")
class TFConcat():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs):
        super().__init__()
        #TODO can be optimzer by watch after node, if conv to be channel last.
        self._axis = node_attribute['axis']
        # use `count` to count how much more for channel-last to channel-first
        count = 0
        for inp in node_inputs:
            if inp in node_weights:
                count -= 1
            elif layout_dict[inp] == Layout.Channel_Last:
                count += 1
            else:
                count -= 1
        
        self._gather = []
        if count < 0:
            # align to Channel_First
            layout_dict[node_outputs[0]] = Layout.Channel_First
            for inp in node_inputs:
                if inp in tensor_grap:
                    if layout_dict[inp] == Layout.Channel_Last:
                        tensor_grap[inp] = dimension_utils.tensor_NDC_to_NCD_format(tensor_grap[inp])
                    self._gather.append(tensor_grap[inp])
                else:
                    self._gather.append(node_weights[inp])
        else:
            # align to Channel_Last
            layout_dict[node_outputs[0]] = Layout.Channel_Last
            self._axis = dimension_utils.channel_to_last_dimension(self._axis)
            for inp in node_inputs:
                if inp in tensor_grap:
                    if layout_dict[inp] != Layout.Channel_Last:
                        tensor_grap[inp] = dimension_utils.tensor_NCD_to_NDC_format(tensor_grap[inp])
                    self._gather.append(tensor_grap[inp])
                else:
                    self._gather.append(dimension_utils.tensor_NCD_to_NDC_format(node_weights[inp]))

    def __call__(self, *args, **kwargs):
        return tf.concat(self._gather, axis=self._axis)

@OPERATOR.register_operator("Reshape")
class TFReshape():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs):
        super().__init__()
        self.out_shape = node_weights[node_inputs[1]]
        self.trans_in = None
        # LOG.info("Reshape will process tensor after change back to NCHW format.")
        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            shape_len = len(tensor_grap[node_inputs[0]].shape)
            self.trans_in = [0, shape_len-1] + [n for n in range(1, shape_len-1)]
        for nop in node_outputs:
            layout_dict[nop] = Layout.Channel_First

    def __call__(self, inputs):
        if self.trans_in:
            inputs = tf.transpose(inputs, perm=self.trans_in)
        inputs = tf.reshape(inputs, shape=self.out_shape)
        return inputs
        
@OPERATOR.register_operator("Flatten")
class TFFlatten():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs)->None:
        super().__init__()
        num_elements = int(tensor_grap[node_inputs[0]].shape.num_elements()/tensor_grap[node_inputs[0]].shape[0])
        input_shape = tensor_grap[node_inputs[0]].shape
        self.flat = tf.keras.layers.Flatten()
        '''
            ensure memory order match, for example:
            onnx = (B, 2, 3, 4).reshape(B, -1)
            tflite = (B, 3, 4, 2).reshape(B, -1)
            we can observe that:
            onnx.shape == tflite.shape, but np.sum(onnx-tflite) != 0
            it's cause memory order of two vars is different, we must make tflite back to onnx by transpose.
            generally, this situation is general one, below is just special situation and most appear in cnn.
            onnx = (B, 512, 1, 1)
            tflite = (B, 1, 1, 512)
            or = (B, 1, 512, 1)
            these memory order are all same.
        '''
        self.perm = None
        if layout_dict[node_inputs[0]] == Layout.Channel_Last and  num_elements != max(input_shape[1:]):
            self.perm = [0, len(input_shape)-1]
            for i in range(len(input_shape)-2):
                self.perm.append(i+1)

    def __call__(self, inputs):
        if self.perm:
            inputs = tf.transpose(inputs, perm=self.perm)
        return self.flat(inputs)

@OPERATOR.register_operator("Split")
class TFSplit():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs)->None:
        super().__init__()
        self.outputs_nums = len(node_outputs)
        self.axis = node_attribute.get("axis", 0)
        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.axis = dimension_utils.channel_to_last_dimension(self.axis)
        split_args = None
        if 'split' in node_attribute:
            split_args = node_attribute['split']
        else:
            assert len(node_inputs) == 2 and node_inputs[1] in node_weights
            split_args = node_weights[node_inputs[1]]
        
        self.indices = []
        start, end = 0, 0
        for i in range(self.outputs_nums):
            end = start + int(split_args[i])
            self.indices.append(tf.keras.backend.arange(start, end, 1))
            start = end

    def __call__(self, inputs):
        return [tf.gather(inputs, indices=indice, axis=self.axis) for indice in self.indices]

@OPERATOR.register_operator("Expand")
class TFExpand():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs)->None:
        super().__init__()
        self.shape = node_weights[node_inputs[1]]
        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.shape = dimension_utils.shape_NCD_to_NDC_format(self.shape)
    def __call__(self, inputs):
        for i in range(len(self.shape)):
            if int(self.shape[i]//inputs.shape[i]) > 1:
                inputs = tf.repeat(inputs, repeats=int(self.shape[i]//inputs.shape[i]), axis=i)
            elif self.shape[i] < inputs.shape[i] and self.shape[i] != 1:
                inputs = tf.repeat(inputs, repeats=int(self.shape[i]), axis=i)
        return inputs
    
@OPERATOR.register_operator("GatherElements")
class TFGatherElements():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs) -> None:
        super().__init__()
        self.axis = node_attribute.get("axis", 1)
        self.indices = None
        if 'indices' in node_attribute:
            self.indices = node_attribute['indices']
            self.indices = dimension_utils.tensor_NCD_to_NDC_format(self.indices)
        elif node_inputs[1] in node_weights:
            self.indices = node_weights[node_inputs[1]]
            self.indices = dimension_utils.tensor_NCD_to_NDC_format(self.indices)
        else:
            self.indices = tensor_grap[node_inputs[1]]
        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.axis = dimension_utils.channel_to_last_dimension(self.axis)
            if len(node_inputs) == 1 or layout_dict[node_inputs[1]] != Layout.Channel_Last:
                self.indices = dimension_utils.tensor_NCD_to_NDC_format(self.indices)

    def gather_elements(self, input_tensor, indices, axis):
        # Get the shape of the input tensor and the indices tensor
        input_shape = tf.shape(input_tensor)
        indices_shape = tf.shape(indices)

        # Create indices for all dimensions
        idx = tf.meshgrid(*[tf.range(s) for s in indices_shape], indexing='ij')
        idx = [tf.cast(i, tf.int64) for i in idx]

        # Replace the axis index with the provided indices
        idx[axis] = tf.cast(indices, tf.int64)

        # Stack indices to form the final gather indices
        gather_indices = tf.stack(idx, axis=-1)

        # Use tf.gather_nd to gather elements
        output_tensor = tf.gather_nd(input_tensor, gather_indices)

        return output_tensor

    def __call__(self, inputs):
        return self.gather_elements(inputs, self.indices, self.axis)
    
@OPERATOR.register_operator("Tile")
class TFTile():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs)->None:
        super().__init__()
        self.repeats = node_attribute['repeats'] if 'repeats' in node_attribute else node_weights[node_inputs[1]]
        if layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.repeats = dimension_utils.shape_NCD_to_NDC_format(self.repeats)

    def __call__(self, inputs):
        for i in range(len(self.repeats)):
            if self.repeats[i] > 1:
                inputs = tf.repeat(inputs, self.repeats[i], axis=i)
        return inputs

@OPERATOR.register_operator("Unsqueeze")
class TFUnsqueeze():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs)->None:
        super().__init__()
        self.axis = node_attribute['axes'] if 'axes' in node_attribute else node_weights[node_inputs[1]]
        if not isinstance(self.axis, int):
            self.axis = int(self.axis[0])
        input_shape = tensor_grap[node_inputs[0]].shape
        if len(input_shape) == 1:
            layout_dict[node_outputs[0]] = Layout.Channel_None
        elif len(input_shape) == 2:
            layout_dict[node_outputs[0]] = Layout.Channel_First
        else:
            layout_dict[node_outputs[0]] = layout_dict[node_inputs[0]]
            if layout_dict[node_inputs[0]] == Layout.Channel_Last:
                self.axis = dimension_utils.channel_to_last_dimension(self.axis)

    def __call__(self, inputs):
        return tf.expand_dims(inputs, self.axis)

@OPERATOR.register_operator("Squeeze")
class TFSqueeze():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs)->None:
        super().__init__()
        self.axis = node_attribute['axes'] if 'axes' in node_attribute else node_weights[node_inputs[1]]
        if not isinstance(self.axis, int):
            self.axis = int(self.axis[0])
        input_shape = tensor_grap[node_inputs[0]].shape
        if len(input_shape) <= 3:
            layout_dict[node_outputs[0]] = Layout.Channel_None
        if len(input_shape) > 2 and layout_dict[node_inputs[0]] == Layout.Channel_Last:
            self.axis = dimension_utils.channel_to_last_dimension(self.axis)

    def __call__(self, inputs):
        return tf.squeeze(inputs, self.axis)

@OPERATOR.register_operator("DepthToSpace")
class TFDepthToSpace():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs)->None:
        super().__init__()
        self.block_size = node_attribute.get("blocksize", 2)
        self.mode = node_attribute.get("mode", "DCR")
        self.channel_last = layout_dict[node_inputs[0]] == Layout.Channel_Last

    def __call__(self, inputs):
        if not self.channel_last:
            inputs = dimension_utils.tensor_NDC_to_NCD_format(inputs)
        if self.mode == "DCR":
            return tf.nn.depth_to_space(inputs, self.block_size)
        elif self.mode == "CRD":
            # help want, native tensorflow is not support CRD mode, this way will generate 5 dims op.
            b, h, w, c = inputs.shape
            inputs = tf.reshape(inputs, [b, h, w, c//(self.block_size * self.block_size), self.block_size, self.block_size])
            inputs = tf.transpose(inputs, perm=[0, 1, 4, 2, 5, 3])
            inputs = tf.reshape(inputs, [b, h*self.block_size, w*self.block_size, c//(self.block_size * self.block_size)])
            return inputs
        else:
            raise KeyError(f"For DepthToSpace, mode must be [DCR, CRD], not {self.mode}")