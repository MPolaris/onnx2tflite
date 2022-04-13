import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import OPERATOR
from . import shape_axis_utils

@OPERATOR.register_operator("BatchNormalization")
class TFBatchNormalization(keras.layers.Layer):
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

    def call(self, inputs):
        return self.bn(inputs)

@OPERATOR.register_operator("Pad")
class TFPad(keras.layers.Layer):
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

    def call(self, inputs):
        return tf.pad(inputs, self.pad, mode=self.model)

@OPERATOR.register_operator("Clip")
class TFClip(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.min = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.max = tensor_grap[node_inputs[2]] if node_inputs[2] in tensor_grap else node_weights[node_inputs[2]]

    def __call__(self, inputs):
        return tf.clip_by_value(inputs, self.min, self.max)

@OPERATOR.register_operator("Add")
class TFAdd(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.t1 = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[0]])
        self.t2 = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[1]])

    def __call__(self, *args, **kwargs):
        return self.t1 + self.t2

@OPERATOR.register_operator("Sub")
class TFSub(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.t1 = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[0]])
        self.t2 = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[1]])

    def __call__(self, *args, **kwargs):
        return self.t1 - self.t2

@OPERATOR.register_operator("Mul")
class TFMul(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.t1 = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[0]])
        self.t2 = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[1]])

    def __call__(self, *args, **kwargs):
        return self.t1 * self.t2

@OPERATOR.register_operator("Div")
class TFDiv(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.t1 = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[0]])
        self.t2 = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else shape_axis_utils.TorchWeights2TF(node_weights[node_inputs[1]])

    def __call__(self, *args, **kwargs):
        return self.t1 / self.t2

@OPERATOR.register_operator("GlobalAveragePool")
class TFGlobalAveragePool(keras.layers.Layer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)

@OPERATOR.register_operator("AveragePool")
class TFAveragePool(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        self.avg_pool = keras.layers.AveragePooling2D(pool_size=node_attribute.get("kernel_shape", [2])[0], 
                                                        strides=node_attribute.get("strides", [1])[0], padding='VALID')
        self.pad = node_attribute.get("pads", None)
        if self.pad is not None:
            self.pad = TFPad(None, None, None, {"pads": self.pad[0], "mode": "constant"})

    def call(self, inputs):
        if self.pad:
            return self.avg_pool(self.pad(inputs))
        else:
            return self.avg_pool(inputs)

@OPERATOR.register_operator("MaxPool")
class TFMaxPool(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        self.max_pool = keras.layers.MaxPool2D(pool_size=node_attribute.get("kernel_shape", [2])[0], 
                                                 strides=node_attribute.get("strides", [1])[0], padding='VALID')
        self.pad = node_attribute.get("pads", None)
        if self.pad is not None:
            self.pad = TFPad(None, None, None, {"pads": self.pad[0], "mode": "constant"})

    def call(self, inputs):
        if self.pad:
            return self.max_pool(self.pad(inputs))
        else:
            return self.max_pool(inputs)

@OPERATOR.register_operator("Gather")
class TFGather(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        self.axis = shape_axis_utils.Torch2TFAxis(node_attribute['axis'])
        self.indices = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=self.axis)

@OPERATOR.register_operator("Transpose")
class TFTranspose(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        if kwargs.get("perm_list"):
            self.perm_list = kwargs.get("perm_list")
        else:
            self.perm_list = shape_axis_utils.TorchShape2TF(node_attribute['perm'])

    def __call__(self, inputs):
        return tf.transpose(inputs, perm=self.perm_list)

@OPERATOR.register_operator("Slice")
class TFSlice(keras.layers.Layer):
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

    def call(self, inputs):
        indices = tf.keras.backend.arange(self.starts, self.ends, step=self.steps)
        return tf.gather(inputs, indices, axis=self.axis)

@OPERATOR.register_operator("Concat")
def TFConcat(tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
    _axis = shape_axis_utils.Torch2TFAxis(node_attribute['axis'])
    _gather = [tensor_grap[x] for x in node_inputs]
    out = tf.concat(_gather, axis=_axis)
    def lambda_func(*args, **kwargs):
        return out
    return lambda_func

@OPERATOR.register_operator("Upsample")
class TFUpsample(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        _, h, w, _ = tensor_grap[node_inputs[0]].shape
        scale = node_weights[node_inputs[1]]

        self.scale = (int(h*scale[2]), int(w*scale[3]))
        if node_attribute.get("mode", "nearest").lower() == 'nearest':
            self.method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        else:
            self.method = tf.image.ResizeMethod.BILINEAR

    def call(self, inputs):
        return tf.image.resize(inputs,  self.scale, method=self.method)

@OPERATOR.register_operator("Resize")
class TFResize(keras.layers.Layer):
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

    def call(self, inputs):
        return tf.image.resize(inputs,  self.scale, method=self.method)

@OPERATOR.register_operator("Reshape")
class TFReshape(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.out_shape = node_weights[node_inputs[1]]
        self.out_shape = shape_axis_utils.TorchShape2TF(self.out_shape)

    def call(self, inputs):
        return tf.reshape(inputs, shape=self.out_shape)

@OPERATOR.register_operator("Flatten")
class TFFlatten(keras.layers.Layer):
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

@OPERATOR.register_operator("Gemm")
class TFGemm(keras.layers.Layer):
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
    def call(self, inputs):
        return self.dense(inputs)

