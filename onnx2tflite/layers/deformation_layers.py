import logging
import tensorflow as tf

from onnx2tflite.utils import OPERATOR, dimension_utils

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
        self._axis = dimension_utils.channel_to_last_dimension(node_attribute['axis'])
        self._gather = [tensor_grap[x] if x in tensor_grap else dimension_utils.tensor_NCD_to_NDC_format(node_weights[x]) for x in node_inputs]

    def __call__(self, *args, **kwargs):
        return tf.concat(self._gather, axis=self._axis)

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
        if num_elements != max(input_shape[1:]):
            self.perm = [0, len(input_shape)-1]
            for i in range(len(input_shape)-2):
                self.perm.append(i+1)

    def __call__(self, inputs):
        if self.perm:
            inputs = tf.transpose(inputs, perm=self.perm)
        return self.flat(inputs)

@OPERATOR.register_operator("Split")
class TFSplit():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.outputs_nums = len(kwargs.get('outputs', [1]))
        self.axis = dimension_utils.channel_to_last_dimension(node_attribute.get("axis", 0))
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
    
@OPERATOR.register_operator("GatherElements")
class TFGatherElements():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        self.axis = node_attribute.get("axis", 1)
        self.axis = dimension_utils.channel_to_last_dimension(self.axis)
        self.indices = None
        if 'indices' in node_attribute:
            self.indices = node_attribute['indices']
            self.indices = dimension_utils.tensor_NCD_to_NDC_format(self.indices)
        elif node_inputs[1] in node_weights:
            self.indices = node_weights[node_inputs[1]]
            self.indices = dimension_utils.tensor_NCD_to_NDC_format(self.indices)
        else:
            self.indices = tensor_grap[node_inputs[1]]

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
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.repeats = node_attribute['repeats'] if 'repeats' in node_attribute else node_weights[node_inputs[1]]
        self.repeats = dimension_utils.shape_NCD_to_NDC_format(self.repeats)

    def __call__(self, inputs):
        for i in range(len(self.repeats)):
            if self.repeats[i] > 1:
                inputs = tf.repeat(inputs, self.repeats[i], axis=i)
        return inputs

@OPERATOR.register_operator("Unsqueeze")
class TFUnsqueeze():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.axis = node_attribute['axes'] if 'axes' in node_attribute else node_weights[node_inputs[1]]
        self.axis = dimension_utils.channel_to_last_dimension(self.axis[0])

    def __call__(self, inputs):
        return tf.expand_dims(inputs, self.axis)

@OPERATOR.register_operator("Squeeze")
class TFSqueeze():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.axis = node_attribute['axes'] if 'axes' in node_attribute else node_weights[node_inputs[1]]
        self.axis = dimension_utils.channel_to_last_dimension(self.axis[0])

    def __call__(self, inputs):
        return tf.squeeze(inputs, self.axis)

@OPERATOR.register_operator("DepthToSpace")
class TFDepthToSpace():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.block_size = node_attribute.get("blocksize", 2)
        self.mode = node_attribute.get("mode", "DCR")

    def __call__(self, inputs):
        if self.mode == "DCR":
            return tf.nn.depth_to_space(inputs, self.block_size)
        elif self.mode == "CRD":
            # help want, native tensorflow is not support CRD mode, this way will generate 5 dims op.
            b, h, w, c = inputs.shape
            tmp = tf.reshape(inputs, [b, h, w, c//(self.block_size * self.block_size), self.block_size, self.block_size])
            tmp = tf.transpose(tmp, perm=[0, 1, 4, 2, 5, 3])
            tmp = tf.reshape(tmp, [b, h*self.block_size, w*self.block_size, c//(self.block_size * self.block_size)])
            return tmp
        else:
            raise KeyError(f"For DepthToSpace, mode must be [DCR, CRD], not {self.mode}")