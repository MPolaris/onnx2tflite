import math
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

from onnx2tflite.utils.definitions import Layout
from onnx2tflite.utils import OPERATOR, intfloat_to_list, dimension_utils

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
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs):
        super().__init__()
        self.epsilon = node_attribute.get("epsilon", 1e-5)
        self.scale = node_weights[node_inputs[1]]
        self.bias = node_weights[node_inputs[2]]
        self.channel_last = layout_dict[node_inputs[0]] == Layout.Channel_Last

    def __call__(self, inputs):
        axes = tuple(range(1, len(inputs.shape)-1)) if self.channel_last else tuple(range(2, len(inputs.shape)))
        mean = tf.reduce_mean(inputs, axis=axes, keepdims=True)
        var = tf.math.reduce_variance(inputs, axis= axes, keepdims=True)
        return self.scale*(inputs - mean)/tf.sqrt(var + self.epsilon) + self.bias

@OPERATOR.register_operator("Pad")
class TFPad():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs):
        super().__init__()
        if node_attribute.get("pads") is not None:
            pads = node_attribute['pads']
        elif node_inputs[1] in node_weights:
            pads = node_weights[node_inputs[1]]
        else:
            pads = tensor_grap[node_inputs[1]]
        self.pad = [[pads[0], pads[4]], [pads[2], pads[6]], [pads[3], pads[7]], [pads[1], pads[5]]]
        self.model = node_attribute.get("mode", "constant").upper()
        self.channel_last = layout_dict[node_inputs[0]] == Layout.Channel_Last
        layout_dict[node_outputs[0]] = Layout.Channel_Last

    def __call__(self, inputs):
        if not self.channel_last:
            inputs = dimension_utils.tensor_NCD_to_NDC_format(inputs)
        return tf.pad(inputs, self.pad, mode=self.model)

@OPERATOR.register_operator("Clip")
class TFClip():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        if "min" in node_attribute:
            self.min = node_attribute.get("min")
        else:
            self.min = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        if "max" in node_attribute:
            self.max = node_attribute.get("max")
        else:
            self.max = tensor_grap[node_inputs[2]] if node_inputs[2] in tensor_grap else node_weights[node_inputs[2]]

    def __call__(self, inputs):
        if float(self.min) == 0 and float(self.max) == 6:
            return tf.nn.relu6(inputs)
        return tf.clip_by_value(inputs, self.min, self.max)

@OPERATOR.register_operator("TFGlobalMaxPool")
class TFGlobalMaxPool():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs) -> None:
        super().__init__()
        self.channel_last = layout_dict[node_inputs[0]] == Layout.Channel_Last

    def __call__(self, inputs):
        if self.channel_last:
            return tf.reduce_max(inputs, axis=[i for i in range(1, len(inputs.shape)-1)], keepdims=True)
        else:
            return tf.reduce_max(inputs, axis=[i for i in range(2, len(inputs.shape))], keepdims=True)

@OPERATOR.register_operator("GlobalAveragePool")
class TFGlobalAveragePool():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs) -> None:
        super().__init__()
        self.channel_last = layout_dict[node_inputs[0]] == Layout.Channel_Last

    def __call__(self, inputs):
        if self.channel_last:
            return tf.reduce_mean(inputs, axis=[i for i in range(1, len(inputs.shape)-1)], keepdims=True)
        else:
            return tf.reduce_mean(inputs, axis=[i for i in range(2, len(inputs.shape))], keepdims=True)

@OPERATOR.register_operator("AveragePool")
class TFAveragePool():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs) -> None:
        super().__init__()
        kernel_shape = intfloat_to_list(node_attribute.get("kernel_shape", [2, 2]), 2)
        strides = intfloat_to_list(node_attribute.get("strides", [1, 1]), 2)
        dilations = intfloat_to_list(node_attribute.get("dilations", [1, 1]), 2)
        ceil_mode = node_attribute.get("ceil_mode", 0)
        pads = intfloat_to_list(node_attribute.get("pads", [0, 0, 0, 0]), 4)

        func = math.floor if ceil_mode == 0 else math.ceil
        
        pad_mode = "SAME"
        input_shape = tensor_grap[node_inputs[0]].shape
        for i in range(len(input_shape)-2):
            pad_shape = pads[i] + pads[i+2]
            onnx_output_shape = func((input_shape[1+i]+pad_shape-((kernel_shape[i]-1)*dilations[i]+1))/strides[i]+1)
            tf_output_shape = math.floor((input_shape[1+i] - kernel_shape[i]) / strides[i]) + 1
            pads[2+i] = max(onnx_output_shape-tf_output_shape, pads[2+i]) # right_down pad
            if pad_mode == "SAME" and onnx_output_shape != input_shape[1+i]:
                pad_mode = "VALID"
        self.max_pool = keras.layers.MaxPool2D(pool_size=kernel_shape, strides=strides, padding=pad_mode)
        
        self.pad = None
        if pad_mode == "VALID" and pads is not None and np.sum(pads) > 0:
            if np.sum(pads) > 0:
                self.pad = keras.layers.ZeroPadding2D(padding=((pads[0], pads[2]), (pads[1], pads[3])))
            
        self.channel_last = layout_dict[node_inputs[0]] == Layout.Channel_Last
        layout_dict[node_outputs[0]] = Layout.Channel_Last

    def __call__(self, inputs):
        if not self.channel_last:
            inputs = dimension_utils.tensor_NCD_to_NDC_format(inputs)
        if self.pad:
            inputs = self.pad(inputs)
        return self.avg_pool(inputs)

@OPERATOR.register_operator("MaxPool")
class TFMaxPool():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs) -> None:
        super().__init__()
        kernel_shape = intfloat_to_list(node_attribute.get("kernel_shape", [2, 2]), 2)
        strides = intfloat_to_list(node_attribute.get("strides", [1, 1]), 2)
        dilations = intfloat_to_list(node_attribute.get("dilations", [1, 1]), 2)
        ceil_mode = node_attribute.get("ceil_mode", 0)
        pads = intfloat_to_list(node_attribute.get("pads", [0, 0, 0, 0]), 4)

        func = math.floor if ceil_mode == 0 else math.ceil

        pad_mode = "SAME"
        input_shape = tensor_grap[node_inputs[0]].shape
        for i in range(len(input_shape)-2):
            pad_shape = pads[i] + pads[i+2]
            onnx_output_shape = func((input_shape[1+i]+pad_shape-((kernel_shape[i]-1)*dilations[i]+1))/strides[i]+1)
            tf_output_shape = math.floor((input_shape[1+i] - kernel_shape[i]) / strides[i]) + 1
            pads[2+i] = max(onnx_output_shape-tf_output_shape, pads[2+i]) # right_down pad
            if pad_mode == "SAME" and onnx_output_shape != input_shape[1+i]:
                pad_mode = "VALID"
        self.max_pool = keras.layers.MaxPool2D(pool_size=kernel_shape, strides=strides, padding=pad_mode)
        
        self.pad = None
        if pad_mode == "VALID" and pads is not None and np.sum(pads) > 0:
            if np.sum(pads) > 0:
                self.pad = keras.layers.ZeroPadding2D(padding=((pads[0], pads[2]), (pads[1], pads[3])))
            
        self.channel_last = layout_dict[node_inputs[0]] == Layout.Channel_Last
        layout_dict[node_outputs[0]] = Layout.Channel_Last

    def __call__(self, inputs):
        if not self.channel_last:
            inputs = dimension_utils.tensor_NCD_to_NDC_format(inputs)
        if self.pad:
            inputs = self.pad(inputs)
        return self.max_pool(inputs)

@OPERATOR.register_operator("Upsample")
class TFUpsample():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs):
        super().__init__()
        _, h, w, _ = tensor_grap[node_inputs[0]].shape
        scale = node_weights[node_inputs[1]]

        self.scale = (int(h*scale[2]), int(w*scale[3]))
        if node_attribute.get("mode", "nearest").lower() == 'nearest':
            self.method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        else:
            self.method = tf.image.ResizeMethod.BILINEAR

        self.channel_last = layout_dict[node_inputs[0]] == Layout.Channel_Last
        layout_dict[node_outputs[0]] = Layout.Channel_Last

    def __call__(self, inputs):
        if not self.channel_last:
            inputs = dimension_utils.tensor_NCD_to_NDC_format(inputs)
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
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs):
        super().__init__()
        self.indices = node_weights[node_inputs[1]]
        self.channle_last = layout_dict[node_inputs[0]] == Layout.Channel_Last
        if node_inputs[2] in tensor_grap:
            self.updates = tensor_grap[node_inputs[2]]
            if self.channle_last:
                self.updates = dimension_utils.tensor_NDC_to_NCD_format(self.updates)
        else:
            self.updates = node_weights[node_inputs[2]]

        layout_dict[node_outputs[0]] = Layout.Channel_First

    def __call__(self, inputs):
        if self.channle_last:
            inputs = dimension_utils.tensor_NDC_to_NCD_format(inputs)
        inputs = tf.tensor_scatter_nd_update(inputs, self.indices, self.updates)
        return inputs

@OPERATOR.register_operator("Resize")
class TFResize():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs):
        super().__init__()
        if node_inputs[-1] in node_weights:
            _, _, nh, nw = node_weights[node_inputs[-1]]
            if len(node_inputs) != 4:
                _, h, w, _ = tensor_grap[node_inputs[0]].shape
                nh, nw = int(h*nh), int(w*nw)
            self.scale = (nh, nw)
        else:
            scales = tensor_grap[node_inputs[0]].shape[1:3]*tensor_grap[node_inputs[2]][2:3]
            self.scale = scales

        if node_attribute.get("mode", "nearest").lower() == 'nearest':
            self.method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        else:
            self.method = tf.image.ResizeMethod.BILINEAR

        self.channel_last = layout_dict[node_inputs[0]] == Layout.Channel_Last
        layout_dict[node_outputs[0]] = Layout.Channel_Last

    def __call__(self, inputs):
        if not self.channel_last:
            inputs = dimension_utils.tensor_NCD_to_NDC_format(inputs)
        return tf.image.resize(inputs,  self.scale, method=self.method)

@OPERATOR.register_operator("Gemm")
class TFGemm():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs) -> None:
        super().__init__()
        if len(node_inputs) > 2:
            weights = [node_weights[node_inputs[1]].T, node_weights[node_inputs[2]]]
        else:
            weights = [node_weights[node_inputs[1]].T]

        self.dense = keras.layers.Dense(weights[0].shape[1],
                                        weights=weights,
                                        use_bias=len(weights)==2)
        
        self.channel_last = layout_dict[node_inputs[0]] == Layout.Channel_Last
        layout_dict[node_outputs[0]] = Layout.Channel_Last

    def __call__(self, inputs):
        if not self.channel_last:
            inputs = dimension_utils.tensor_NCD_to_NDC_format(inputs)
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
    
@OPERATOR.register_operator("TopK")
class TFTopK():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:

        self.axis = node_attribute.get("axis", -1)
        self.largest = node_attribute.get("largest", 1)
        self.sorted = bool(node_attribute.get("sorted", 1))
        self.K = node_attribute.get('K') if len(node_inputs)==1 else node_weights[node_inputs[1]][0]

    def __call__(self, inputs):
        res = tf.math.top_k(inputs, k=self.K, sorted=self.sorted)
        return [res[0], res[1]] 
    
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
