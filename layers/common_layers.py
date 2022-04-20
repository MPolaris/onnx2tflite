import numpy as np
import tensorflow as tf
import logging
from tensorflow import keras

from . import OPERATOR
from . import shape_axis_utils

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

@OPERATOR.register_operator("Pad")
class TFPad():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        if node_attribute.get("pads") is not None:
            pad = np.max(node_attribute['pads'])
        elif node_inputs[1] in node_weights:
            pad = np.max(node_weights[node_inputs[1]])
        else:
            pad = np.max(tensor_grap[node_inputs[1]])
        self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        self.model = node_attribute['mode'].upper()

    def __call__(self, inputs):
        return tf.pad(inputs, self.pad, mode=self.model)

@OPERATOR.register_operator("Clip")
class TFClip():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.min = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.max = tensor_grap[node_inputs[2]] if node_inputs[2] in tensor_grap else node_weights[node_inputs[2]]

    def __call__(self, inputs):
        return tf.clip_by_value(inputs, self.min, self.max)

@OPERATOR.register_operator("GlobalAveragePool")
class TFGlobalAveragePool():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inputs):
        return tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)

@OPERATOR.register_operator("AveragePool")
class TFAveragePool():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        self.avg_pool = keras.layers.AveragePooling2D(pool_size=node_attribute.get("kernel_shape", [2])[0], 
                                                        strides=node_attribute.get("strides", [1])[0], padding='VALID')
        self.pad = node_attribute.get("pads", None)
        if self.pad is not None:
            self.pad = TFPad(None, None, None, {"pads": self.pad[0], "mode": "constant"})

    def __call__(self, inputs):
        if self.pad:
            return self.avg_pool(self.pad(inputs))
        else:
            return self.avg_pool(inputs)

@OPERATOR.register_operator("MaxPool")
class TFMaxPool():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        # TODO 这个操作会导致转换误差增大，估计是pad的问题导致的。
        self.max_pool = keras.layers.MaxPool2D(pool_size=node_attribute.get("kernel_shape", [2])[0], 
                                                 strides=node_attribute.get("strides", [1])[0], padding='VALID')
        self.pad = node_attribute.get("pads", None)
        if self.pad is not None:
            self.pad = TFPad(None, None, None, {"pads": self.pad[0], "mode": "constant"})
            # self.pad = keras.layers.ZeroPadding2D(padding=((self.pad[0], self.pad[1]), (self.pad[2], self.pad[3])))

    def __call__(self, inputs):
        if self.pad:
            return self.max_pool(self.pad(inputs))
        else:
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

