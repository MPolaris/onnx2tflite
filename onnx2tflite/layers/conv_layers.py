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
from onnx2tflite.utils.op_registry import OPERATOR
from onnx2tflite.utils.definitions import Layout
from onnx2tflite.utils.dimension_utils import tensor_NCD_to_NDC_format as NCD2NDC

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
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs) -> None:
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
                                                    dilation_rate=dilations)
        if pads is not None and max(pads) != 0:
            padding = None
            if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
                padding = (pads[0], pads[1])
            elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
                padding = ((pads[0], pads[2]), (pads[1], pads[3]))
            self.pad = keras.layers.Cropping2D(padding)

        for nop in node_outputs:
            layout_dict[nop] = Layout.Channel_Last

        self.need_trans = layout_dict[node_inputs[0]] != Layout.Channel_Last

    def __call__(self, inputs):
        if self.need_trans:
            inputs = NCD2NDC(inputs)
        inputs = self.conv(inputs)
        if self.pad:
            inputs = self.pad(inputs)
        return inputs

@OPERATOR.register_operator("Conv")
class Convlution():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs) -> None:
        super().__init__()
        out_channel, in_channel = node_weights[node_inputs[1]].shape[:2]
        dilations, group = node_attribute.get('dilations', 1), node_attribute.get('group', 1)
        pads = node_attribute['pads'] if "pads" in node_attribute else None
        kernel_shape, strides = node_attribute.get('kernel_shape', 1), node_attribute.get('strides', 1)

        weights = node_weights[node_inputs[1]] if node_inputs[1] in node_weights else tensor_grap[node_inputs[1]]
        out_channel, in_channel = weights.shape[:2]

        channel_sequence = [2+i for i in range(len(weights.shape)-2)] + [1, 0]
        weights = weights.transpose(*channel_sequence)

        bias = None
        if len(node_inputs) == 3:
            bias = node_weights[node_inputs[2]] if node_inputs[2] in node_weights else tensor_grap[node_inputs[2]]

        if group == 1:
            self.conv = TFConv(in_channel, out_channel, kernel_shape, strides, dilations, pads, weights, bias)
        elif group == out_channel:
            self.conv = TFDepthwiseConv(kernel_shape, strides, dilations, pads, weights, bias)
        else:
            if USE_NATIVE_GROUP_CONV:
                self.conv = TFConv(in_channel, out_channel, kernel_shape, strides, dilations, pads, weights, bias, group=group)
                LOG.warning(f"Group Convolution is detected, using native method, only supported tflite version >= 2.9, \
                                if compatibility error occurs and please make USE_NATIVE_GROUP_CONV=False!")
            else:
                self.conv = TFGroupConv(in_channel, out_channel, kernel_shape, strides, dilations, pads, weights, bias, group=group)

        for nop in node_outputs:
            layout_dict[nop] = Layout.Channel_Last

        self.need_trans = layout_dict[node_inputs[0]] != Layout.Channel_Last

    def __call__(self, inputs):
        if self.need_trans:
            inputs = NCD2NDC(inputs)
        return self.conv(inputs)

class TFConv():
    # Standard convolution
    def __init__(self, in_channel_num, out_channel_num, kernel_size=1, 
                        strides=1, dilations=1, pads=None, weights=None, bias=None, group=1):
        super().__init__()

        if len(weights.shape) == 3:
            self.conv1d_init(in_channel_num, out_channel_num, kernel_size, strides, dilations, pads, weights, bias, group)
        elif len(weights.shape) == 4:
            self.conv2d_init(in_channel_num, out_channel_num, kernel_size, strides, dilations, pads, weights, bias, group)
        elif len(weights.shape) == 5:
            self.conv3d_init(in_channel_num, out_channel_num, kernel_size, strides, dilations, pads, weights, bias, group)
        else:
            raise NotImplementedError(f"Conv{len(weights.shape)-2}d is not implemented")

    def conv1d_init(self, in_channel_num, out_channel_num, kernel_size=1, 
                    strides=1, dilations=1, pads=None, weights=None, bias=None, group=1):
        self.pad =None
        if pads is not None and max(pads) == 1 and max(strides) == 1:
            self.conv = keras.layers.Conv1D(
                out_channel_num, kernel_size, strides, "SAME", use_bias=False if bias is None else True,
                weights=[weights] if bias is None else [weights, bias],
                dilation_rate=dilations, groups=group)
        else:
            self.conv = keras.layers.Conv1D(
                out_channel_num, kernel_size, strides, "VALID", use_bias=False if bias is None else True,
                weights=[weights] if bias is None else [weights, bias],
                dilation_rate=dilations, groups=group)
            if pads is not None and max(pads) != 0:
                self.pad = keras.layers.ZeroPadding1D(padding=pads)

    def conv2d_init(self, in_channel_num, out_channel_num, kernel_size=1, 
                        strides=1, dilations=1, pads=None, weights=None, bias=None, group=1):
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
                weights=[weights] if bias is None else [weights, bias],
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

    def conv3d_init(self, in_channel_num, out_channel_num, kernel_size=1, 
                    strides=1, dilations=1, pads=None, weights=None, bias=None, group=1):
        raise NotImplementedError("Conv3d is not implemented")

    def __call__(self, inputs):
        if self.pad:
            inputs = self.pad(inputs)
        return self.conv(inputs)

class TFGroupConv():
    '''
        Group Convolution, using split method to implement, not native.
    '''
    def __init__(self, in_channel_num, out_channel_num, kernel_size=1, 
                        strides=1, dilations=1, pads=None, weights=None, bias=None, group=1):
        super().__init__()

        if len(weights.shape) == 3:
            self.groupconv1d_init(in_channel_num, out_channel_num, kernel_size, strides, dilations, pads, weights, bias, group)
        elif len(weights.shape) == 4:
            self.groupconv2d_init(in_channel_num, out_channel_num, kernel_size, strides, dilations, pads, weights, bias, group)
        else:
            raise NotImplementedError(f"GroupConv{len(weights.shape)-2}d is not implemented")

    def groupconv1d_init(self, in_channel_num, out_channel_num, kernel_size=1, 
                        strides=1, dilations=1, pads=None, weights=None, bias=None, group=1):
        self.cin = in_channel_num
        self.groups = group
        out_channel_num = int(out_channel_num//group)
        self.convs = []
        for i in range(group):
            if pads is not None and max(pads) == 1 and max(strides) == 1:
                self.convs.append(keras.layers.Conv1D(
                                out_channel_num, kernel_size, strides, 'SAME', use_bias=False if bias is None else True,
                                dilation_rate=dilations,
                                weights=[weights[:, :, i*out_channel_num:(i+1)*out_channel_num]] if bias is None else [weights[:, :, i*out_channel_num:(i+1)*out_channel_num], bias[i*out_channel_num:(i+1)*out_channel_num]]))
            else:
                self.convs.append(keras.layers.Conv1D(
                                    out_channel_num, kernel_size, strides, 'VALID', use_bias=False if bias is None else True,
                                    dilation_rate=dilations,
                                    weights=[weights[:, :, i*out_channel_num:(i+1)*out_channel_num]] if bias is None else [weights[:, :, i*out_channel_num:(i+1)*out_channel_num], bias[i*out_channel_num:(i+1)*out_channel_num]]))
                self.pad =None
                if pads is not None and (max(pads) != 0 and not (max(pads) == 1 and max(strides) == 1)):
                    self.pad = keras.layers.ZeroPadding1D(padding=pads)

    def groupconv2d_init(self, in_channel_num, out_channel_num, kernel_size=1, 
                        strides=1, dilations=1, pads=None, weights=None, bias=None, group=1):
        if isinstance(dilations, int):
            dilations = (dilations, dilations)
        if isinstance(strides, int):
            strides = (strides, strides)
        if dilations[0] != 1 and strides[0] != 1:
            raise Exception("Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.")
        self.cin = in_channel_num
        self.groups = group
        out_channel_num = int(out_channel_num//group)

        self.convs = []
        for i in range(group):
            if pads is not None and max(pads) == 1 and max(strides) == 1:
                self.convs.append(keras.layers.Conv2D(
                                out_channel_num, kernel_size, strides, 'SAME', use_bias=False if bias is None else True,
                                dilation_rate=dilations,
                                weights=[weights[:, :, :, i*out_channel_num:(i+1)*out_channel_num]] if bias is None else [weights[:, :, :, i*out_channel_num:(i+1)*out_channel_num], bias[i*out_channel_num:(i+1)*out_channel_num]]))
            else:
                self.convs.append(keras.layers.Conv2D(
                                    out_channel_num, kernel_size, strides, 'VALID', use_bias=False if bias is None else True,
                                    dilation_rate=dilations,
                                    weights=[weights[:, :, :, i*out_channel_num:(i+1)*out_channel_num]] if bias is None else [weights[:, :, :, i*out_channel_num:(i+1)*out_channel_num], bias[i*out_channel_num:(i+1)*out_channel_num]]))
                self.pad =None
                if pads is not None and (max(pads) != 0 and not (max(pads) == 1 and max(strides) == 1)):
                    padding = None
                    if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
                        padding = (pads[0], pads[1])
                    elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
                        padding = ((pads[0], pads[2]), (pads[1], pads[3]))
                    self.pad = keras.layers.ZeroPadding2D(padding=padding)

    def __call__(self, inputs):
        if self.pad is not None:
            inputs = self.pad(inputs)
        outs = []
        in_s = tf.split(inputs, num_or_size_splits=self.groups, axis=-1)
        for i in range(self.groups):
            outs.append(self.convs[i](in_s[i]))
        outs = tf.concat(outs, axis=-1)
        return outs

class TFDepthwiseConv():
    # Depthwise Convolution, group = 1
    def __init__(self, kernel_size=1, strides=1, dilations=1, pads=None, weights=None, bias=None) -> None:
        super().__init__()
        if len(weights.shape) == 3:
            weights = weights.transpose(0, 2, 1)
            self.dwconv1d_init(kernel_size, strides, dilations, pads, weights, bias)
        elif len(weights.shape) == 4:
            weights = weights.transpose(0, 1, 3, 2)
            self.dwconv2d_init(kernel_size, strides, dilations, pads, weights, bias)
        else:
            raise NotImplementedError(f"DepthwiseConv{len(weights.shape)-2}d is not implemented")

    def dwconv1d_init(self, kernel_size=1, strides=1, dilations=1, pads=None, weights=None, bias=None):
        self.pad =None
        if pads is not None and max(pads) == 1 and max(strides) == 1:
            self.conv = keras.layers.DepthwiseConv1D(
                kernel_size, strides, "SAME", use_bias=False if bias is None else True,
                weights=[weights] if bias is None else [weights, bias],
                dilation_rate=dilations,
                activation=None,
                kernel_initializer='zeros',
                bias_initializer='zeros'
            )
        else:
            self.conv = keras.layers.DepthwiseConv1D(
                kernel_size, strides, "VALID", use_bias=False if bias is None else True,
                weights=[weights] if bias is None else [weights, bias],
                dilation_rate=dilations,
                activation=None,
                kernel_initializer='zeros',
                bias_initializer='zeros'
            )
            if pads is not None and max(pads) != 0:
                self.pad = keras.layers.ZeroPadding1D(padding=pads)

    def dwconv2d_init(self, kernel_size=1, strides=1, dilations=1, pads=None, weights=None, bias=None):
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