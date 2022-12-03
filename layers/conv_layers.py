'''
    Author: MPolaris && yutaka329 && lkdci

    Thanks for yutaka329 with your pad tricks.
    https://github.com/MPolaris/onnx2tflite/issues/5

    Thanks for lkdci with your native method of group conv
    https://github.com/MPolaris/onnx2tflite/issues/19 
'''
import logging
import tensorflow as tf
from tensorflow import keras
from . import OPERATOR

LOG = logging.getLogger("convolution_layers :")

# Whether to implement grouped convolution using the native `keras.layers.Conv2D` class with groups !=1 argument.
# This implementation is supported only with tflite version >= 2.9.
# If set to `False`, the grouped convolution is built using regular conv per group then concatenated as a workaround
# to support older version of tflite.
# Using the native keras implementation results in a simplified tflite graph and supposed to run faster.
# See https://github.com/MPolaris/onnx2tflite/issues/19 for more details.
USE_NATIVE_GROUP_CONV = False

@OPERATOR.register_operator("ConvTranspose")
class TFConvTranspose():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        # out_channel, in_channel = node_weights[node_inputs[1]].shape[:2]
        dilations, group = node_attribute.get('dilations', 1), node_attribute.get('group', 1)
        pads = node_attribute['pads'] if "pads" in node_attribute else None
        kernel_shape, strides = node_attribute.get('kernel_shape', 1), node_attribute.get('strides', 1)

        weights = node_weights[node_inputs[1]].transpose(2,3,1,0)
        bias = node_weights[node_inputs[2]] if len(node_inputs) == 3 else None
        height, width, n_filters, channels = weights.shape
        self.pad = None
        self.conv = keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=(height, width), strides=strides, padding='VALID', use_bias=False if bias is None else True,
                                                    weights=[weights] if bias is None else [weights, bias],
                                                    output_padding=0,
                                                    dilation_rate=dilations)
        if pads is not None and max(pads) != 0:
            LOG.warning("ConvTranspose with pad will lead output error to bigger, please check it out.")
            assert(len(pads) == 2 or (pads[2] == pads[0] and pads[3] == pads[1]))
            self.pad = keras.layers.Cropping2D(pads[:2])

    def __call__(self, inputs):
        if self.pad:
            inputs = self.pad(inputs)
        inputs = self.conv(inputs)
        return inputs

@OPERATOR.register_operator("Conv")
class Convlution():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        out_channel, in_channel = node_weights[node_inputs[1]].shape[:2]
        dilations, group = node_attribute.get('dilations', 1), node_attribute.get('group', 1)
        pads = node_attribute['pads'] if "pads" in node_attribute else None
        kernel_shape, strides = node_attribute.get('kernel_shape', 1), node_attribute.get('strides', 1)

        weights = node_weights[node_inputs[1]].transpose(2,3,1,0)
        bias = node_weights[node_inputs[2]] if len(node_inputs) == 3 else None
        if group == 1:
            self.conv = TFConv(in_channel, out_channel, kernel_shape, strides, dilations, pads, weights, bias)
        elif group == out_channel:
            weights = weights.transpose(0, 1, 3, 2)
            self.conv = TFDepthwiseConv2D(kernel_shape, strides, dilations, pads, weights, bias)
        else:
            if USE_NATIVE_GROUP_CONV:
                self.conv = TFConv(in_channel, out_channel, kernel_shape, strides, dilations, pads, weights, bias,
                                   group=group)
                LOG.warning(f"Group Convolution is detected, using native method, only supported tflite version >= 2.9, \
                                if compatibility error occurs and please make USE_NATIVE_GROUP_CONV=False!")
            else:
                self.conv = TFGroupConv(in_channel, out_channel, kernel_shape, strides, dilations, pads, group, weights,
                                        bias)
    
    def __call__(self, inputs):
        return self.conv(inputs)

class TFConv():
    # Standard convolution
    def __init__(self, in_channel_num, out_channel_num, kernel_size=1, 
                        strides=1, dilations=1, pads=None, weights=None, bias=None, group=1):
        super().__init__()

        if isinstance(dilations, int):
            dilations = (dilations, dilations)
        if isinstance(strides, int):
            strides = (strides, strides)
        if dilations[0] != 1 and strides[0] != 1:
            raise Exception("Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.")

        self.pad =None
        if pads is not None and max(pads) == 1 and max(strides) == 1:
            self.conv = keras.layers.Conv2D(
                out_channel_num, kernel_size, strides, "SAME", use_bias=False if bias is None else True,
                kernel_initializer=keras.initializers.Constant(weights),
                bias_initializer='zeros' if bias is None else keras.initializers.Constant(bias),
                dilation_rate=dilations, groups=group)
        else:
            self.conv = keras.layers.Conv2D(
                out_channel_num, kernel_size, strides, "VALID", use_bias=False if bias is None else True,
                weights=[weights] if bias is None else [weights, bias],
                dilation_rate=dilations, groups=group)
            if pads is not None and max(pads) != 0:
                padding = None
                if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
                    padding = (pads[0], pads[1])
                elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
                    padding = ((pads[0], pads[2]), (pads[1], pads[3]))
                self.pad = keras.layers.ZeroPadding2D(padding=padding)

    def __call__(self, inputs):
        if self.pad:
            inputs = self.pad(inputs)
        return self.conv(inputs)

class TFGroupConv():
    '''
        Group Convolution, using split method to implement, not native.
    '''
    def __init__(self, in_channel_num, out_channel_num, kernel_size=1, 
                        strides=1, dilations=1, pads=None, groups=1, weights=None, bias=None):
        super().__init__()
        if isinstance(dilations, int):
            dilations = (dilations, dilations)
        if isinstance(strides, int):
            strides = (strides, strides)
        if dilations[0] != 1 and strides[0] != 1:
            raise Exception("Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.")
        self.cin = in_channel_num
        self.groups = groups
        out_channel_num = int(out_channel_num//groups)
        self.pad =None
        if pads is not None and (max(pads) != 0 and not (max(pads) == 1 and max(strides) == 1)):
            padding = None
            if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
                padding = (pads[0], pads[1])
            elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
                padding = ((pads[0], pads[2]), (pads[1], pads[3]))
            self.pad = keras.layers.ZeroPadding2D(padding=padding)
        
        self.convs = []
        for i in range(groups):
            if pads is not None and max(pads) == 1 and max(strides) == 1:
                self.convs.append(keras.layers.Conv2D(
                                out_channel_num, kernel_size, strides, 'SAME', use_bias=False if bias is None else True,
                                dilation_rate=dilations,
                                weights=[weights] if bias is None else [weights, bias]))
            else:
                self.convs.append(keras.layers.Conv2D(
                                    out_channel_num, kernel_size, strides, 'VALID', use_bias=False if bias is None else True,
                                    dilation_rate=dilations,
                                    weights=[weights] if bias is None else [weights, bias]))

    def __call__(self, inputs):
        if self.pad is not None:
            inputs = self.pad(inputs)
        outs = []
        in_s = tf.split(inputs, num_or_size_splits=self.groups, axis=-1)
        for i in range(self.groups):
            outs.append(self.convs[i](in_s[i]))
        outs = tf.concat(outs, axis=-1)
        return outs

class TFDepthwiseConv2D():
    # Depthwise Convolution, group = 1
    def __init__(self, kernel_size=1, strides=1, dilations=1, pads=None, weights=None, bias=None) -> None:
        super().__init__()
        if isinstance(dilations, int):
            dilations = (dilations, dilations)
        if isinstance(strides, int):
            strides = (strides, strides)

        self.pad =None
        if pads is not None and max(pads) == 1 and max(strides) == 1:
            self.conv = keras.layers.DepthwiseConv2D(
                kernel_size, strides, "SAME", use_bias=False if bias is None else True,
                weights=[weights] if bias is None else [weights, bias],
                dilation_rate=dilations,
                activation=None,
                kernel_initializer='zeros',
                bias_initializer='zeros'
            )
        else:
            self.conv = keras.layers.DepthwiseConv2D(
                kernel_size, strides, "VALID", use_bias=False if bias is None else True,
                weights=[weights] if bias is None else [weights, bias],
                dilation_rate=dilations,
                activation=None,
                kernel_initializer='zeros',
                bias_initializer='zeros'
            )
            if pads is not None and max(pads) != 0:
                padding = None
                if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
                    padding = (pads[0], pads[1])
                elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
                    padding = ((pads[0], pads[2]), (pads[1], pads[3]))
                self.pad = keras.layers.ZeroPadding2D(padding=padding)
                
    def __call__(self, inputs):
        if self.pad:
            inputs = self.pad(inputs)
        return self.conv(inputs)