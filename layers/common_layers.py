import math
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import OPERATOR

LOG = logging.getLogger("common_layers :")

@OPERATOR.register_operator("BatchNormalization")
class TFBatchNormalization():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        epsilon = node_attribute.get("epsilon", 1e-5)
        momentum = node_attribute.get("momentum", 0.9)

        self.bn = keras.layers.BatchNormalization(
            gamma_initializer=keras.initializers.Constant(node_weights[node_inputs[1]]),
            beta_initializer=keras.initializers.Constant(node_weights[node_inputs[2]]),
            moving_mean_initializer=keras.initializers.Constant(node_weights[node_inputs[3]]),
            moving_variance_initializer=keras.initializers.Constant(node_weights[node_inputs[4]]),
            epsilon=epsilon,
            momentum=momentum)

    def __call__(self, inputs):
        return self.bn(inputs)

@OPERATOR.register_operator("InstanceNormalization")
class TFInstanceNormalization():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.epsilon = node_attribute.get("epsilon", 1e-5)
        self.scale = node_weights[node_inputs[1]]
        self.bias = node_weights[node_inputs[2]]

    def __call__(self, inputs):
        axes = tuple(range(1, len(inputs.shape)-1))
        mean = tf.reduce_mean(inputs, axis=axes, keepdims=True)
        var = tf.math.reduce_variance(inputs, axis= axes, keepdims=True)
        return self.scale*(inputs - mean)/tf.sqrt(var + self.epsilon) + self.bias

@OPERATOR.register_operator("Pad")
class TFPad():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        if node_attribute.get("pads") is not None:
            pads = node_attribute['pads']
        elif node_inputs[1] in node_weights:
            pads = node_weights[node_inputs[1]]
        else:
            pads = tensor_grap[node_inputs[1]]
        self.pad = [[pads[0], pads[4]], [pads[2], pads[6]], [pads[3], pads[7]], [pads[1], pads[5]]]
        self.model = node_attribute.get("mode", "constant").upper()

    def __call__(self, inputs):
        return tf.pad(inputs, self.pad, mode=self.model)

@OPERATOR.register_operator("Clip")
class TFClip():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.min = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.max = tensor_grap[node_inputs[2]] if node_inputs[2] in tensor_grap else node_weights[node_inputs[2]]

    def __call__(self, inputs):
        if float(self.min) == 0 and float(self.max) == 6:
            return tf.nn.relu6(inputs)
        return tf.clip_by_value(inputs, self.min, self.max)

@OPERATOR.register_operator("TFGlobalMaxPool")
class TFGlobalMaxPool():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return tf.reduce_max(inputs, axis=[i for i in range(1, len(inputs.shape)-1)], keepdims=True)

@OPERATOR.register_operator("GlobalAveragePool")
class TFGlobalAveragePool():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return tf.reduce_mean(inputs, axis=[i for i in range(1, len(inputs.shape)-1)], keepdims=True)

@OPERATOR.register_operator("AveragePool")
class TFAveragePool():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        kernel_shape = node_attribute.get("kernel_shape", [2, 2])
        strides = strides=node_attribute.get("strides", [1, 1])
        ceil_mode = node_attribute.get("ceil_mode", 0)
        pads = node_attribute.get("pads", [0, 0, 0, 0])

        inputshape = tensor_grap[node_inputs[0]].shape
        half_shape = True
        for i in range(len(inputshape)-2):
            if ceil_mode == 0:
                half_shape = half_shape and math.floor((inputshape[1+i]+pads[i]-((kernel_shape[i]-1)*1+1))/strides[i]+1) * 2 == inputshape[1+i]
            else:
                half_shape = half_shape and math.ceil((inputshape[1+i]+pads[i]-((kernel_shape[i]-1)*1+1))/strides[i]+1) * 2 == inputshape[1+i]

        half_shape = half_shape and np.sum(pads) == 0
        pad_mode = "SAME" if half_shape else "VALID" 
        self.avg_pool = keras.layers.AveragePooling2D(pool_size=node_attribute.get("kernel_shape", [2])[0], 
                                                        strides=node_attribute.get("strides", [1])[0], padding=pad_mode)
        
        self.pad = None
        if not half_shape and pads is not None and np.sum(pads) > 0:
            self.pad = keras.layers.ZeroPadding2D(padding=((pads[0], pads[2]), (pads[1], pads[3])))

    def __call__(self, inputs):
        if self.pad:
            inputs = self.pad(inputs)
        return self.avg_pool(inputs)

@OPERATOR.register_operator("MaxPool")
class TFMaxPool():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        kernel_shape = node_attribute.get("kernel_shape", [2, 2])
        strides = strides=node_attribute.get("strides", [1, 1])
        ceil_mode = node_attribute.get("ceil_mode", 0)
        pads = node_attribute.get("pads", [0, 0, 0, 0])

        inputshape = tensor_grap[node_inputs[0]].shape
        half_shape = True
        for i in range(len(inputshape)-2):
            if ceil_mode == 0:
                half_shape = half_shape and math.floor((inputshape[1+i]+pads[i]-((kernel_shape[i]-1)*1+1))/strides[i]+1) * 2 == inputshape[1+i]
            else:
                half_shape = half_shape and math.ceil((inputshape[1+i]+pads[i]-((kernel_shape[i]-1)*1+1))/strides[i]+1) * 2 == inputshape[1+i]

        half_shape = half_shape and np.sum(pads) == 0
        pad_mode = "SAME" if half_shape else "VALID" 
        self.max_pool = keras.layers.MaxPool2D(pool_size=node_attribute.get("kernel_shape", [2])[0], 
                                                 strides=node_attribute.get("strides", [1])[0], padding=pad_mode)
        
        self.pad = None
        if not half_shape and pads is not None and np.sum(pads) > 0:
            self.pad = keras.layers.ZeroPadding2D(padding=((pads[0], pads[2]), (pads[1], pads[3])))
            

    def __call__(self, inputs):
        if self.pad:
            inputs = self.pad(inputs)
        return self.max_pool(inputs)

@OPERATOR.register_operator("Upsample")
class TFUpsample():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        _, h, w, _ = tensor_grap[node_inputs[0]].shape
        scale = node_weights[node_inputs[1]]

        self.scale = (int(h*scale[2]), int(w*scale[3]))
        if node_attribute.get("mode", "nearest").lower() == 'nearest':
            self.method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        else:
            self.method = tf.image.ResizeMethod.BILINEAR

    def __call__(self, inputs):
        return tf.image.resize(inputs,  self.scale, method=self.method)

@OPERATOR.register_operator("Constant")
class TFConstant():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.val = node_attribute['value']

    def __call__(self, *args, **kwargs):
        return self.val

@OPERATOR.register_operator("ScatterND")
class TFScatterND():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.indices = node_weights[node_inputs[1]]
        shape_len = len(tensor_grap[node_inputs[0]].shape)
        self.trans_in = [0, shape_len-1] + [n for n in range(1, shape_len-1)]
        self.trans_out = [0] + [n for n in range(2, shape_len)] + [1]
        if node_inputs[2] in tensor_grap:
            self.updates = tf.transpose(tensor_grap[node_inputs[2]], perm=self.trans_in)
        else:
            self.updates = node_weights[node_inputs[2]]

    def __call__(self, inputs):
        inputs = tf.transpose(inputs, perm=self.trans_in)
        inputs = tf.tensor_scatter_nd_update(inputs, self.indices, self.updates)
        inputs = tf.transpose(inputs, perm=self.trans_out)
        return inputs

@OPERATOR.register_operator("Resize")
class TFResize():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        if len(node_inputs) == 4:
            # 从sizes取
            _, _, nh, nw = node_weights[node_inputs[3]]
        else:
            # 从scales取
            _, _, nh, nw = node_weights[node_inputs[2]]
            _, h, w, _ = tensor_grap[node_inputs[0]].shape
            nh, nw = int(h*nh), int(w*nw)
        
        self.scale = (nh, nw)
        if node_attribute.get("mode", "nearest").lower() == 'nearest':
            self.method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        else:
            self.method = tf.image.ResizeMethod.BILINEAR

    def __call__(self, inputs):
        return tf.image.resize(inputs,  self.scale, method=self.method)

@OPERATOR.register_operator("Gemm")
class TFGemm():
    '''
        全连接函数, torch.linear, tf.layers.dense, keras.layers.Dense
    '''
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        weights = node_weights[node_inputs[1]].T
        bias = node_weights[node_inputs[2]] if len(node_inputs) > 2 else None
        weights = weights
        if bias is None:
            self.dense = keras.layers.Dense(weights.shape[1],
                                            input_shape=(weights.shape[0],),
                                            activation=None,
                                            use_bias=False,
                                            kernel_initializer=keras.initializers.Constant(weights))
        else:
            bias = bias[None, ...]
            self.dense = keras.layers.Dense(weights.shape[1],
                                            input_shape=(weights.shape[0],),
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer=keras.initializers.Constant(weights),
                                            bias_initializer=keras.initializers.Constant(bias))
    def __call__(self, inputs):
        return self.dense(inputs)

@OPERATOR.register_operator("Identity")
class TFIdentity():
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs):
        return inputs

@OPERATOR.register_operator("Dropout")
class TFDropout():
    '''
        Dropout will be ignored in deployment.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs):
        return inputs

@OPERATOR.register_operator("Cast")
class TFCast():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.cast_to = int(node_attribute.get("to", 1))
        assert self.cast_to > 0 and self.cast_to < 12, f"Unknown cast type [{self.cast_to}]"
        self.np_cast_map = {
                1: np.float32,
                2: np.uint8,
                3: np.int8,
                5: np.int16,
                6: np.int32,
                7: np.int64,
                9: np.bool_,
                10: np.float16,
                11: np.double,
            }
        self.tf_cast_map = {
                1: tf.float32,
                2: tf.uint8,
                3: tf.int8,
                5: tf.int16,
                6: tf.int32,
                7: tf.int64,
                9: tf.bool,
                10: tf.float16,
                11: tf.double,
            }

    def __call__(self, inputs):
        if isinstance(inputs, list):
            for i in range(len(inputs)):
                if isinstance(inputs[i], np.ndarray) or isinstance(inputs[i], np.generic):
                    inputs[i] = self.np_cast_map[self.cast_to](inputs[i])
                else:
                    inputs[i] = tf.cast(input[i], dtype=self.tf_cast_map[self.cast_to])
        else:
            if isinstance(inputs, np.ndarray) or isinstance(inputs, np.generic):
                inputs = self.np_cast_map[self.cast_to](inputs)
            else:
                inputs = tf.cast(inputs, dtype=self.tf_cast_map[self.cast_to])

        return inputs
