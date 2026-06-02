"""Convolution operators: Conv, ConvTranspose, DepthwiseConv."""
import numpy as np
from tensorflow.lite.python.schema_py_generated import (
    BuiltinOperator as Op,
    Conv2DOptionsT,
    DepthwiseConv2DOptionsT,
    TransposeConvOptionsT,
)
from onnx2tflite.components.tflite_ir.builder import Layout
from onnx2tflite.components.tflite_ir.op_mapping import _register


def _decode_conv_attrs(node):
    """Extract common conv attributes from ONNX node."""
    attrs = {}
    for attr in node.attribute:
        if attr.name == "kernel_shape":
            attrs["kernel"] = list(attr.ints)
        elif attr.name == "strides":
            attrs["strides"] = list(attr.ints)
        elif attr.name == "dilations":
            attrs["dilations"] = list(attr.ints)
        elif attr.name == "pads":
            attrs["pads"] = list(attr.ints)
        elif attr.name == "group":
            attrs["group"] = attr.i
        elif attr.name == "auto_pad":
            attrs["auto_pad"] = attr.s.decode() if hasattr(attr, 's') else "NOTSET"
    attrs.setdefault("kernel", [1, 1])
    attrs.setdefault("strides", [1, 1])
    attrs.setdefault("dilations", [1, 1])
    attrs.setdefault("pads", [0, 0, 0, 0])
    attrs.setdefault("group", 1)
    attrs.setdefault("auto_pad", "NOTSET")
    return attrs


def _conv_padding(pads, strides, kernel, auto_pad):
    """Determine TFLite padding enum: 0=SAME, 1=VALID."""
    if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
        return 0  # SAME
    if auto_pad == "VALID":
        return 1
    if sum(pads) == 0:
        return 1  # VALID
    # Check if pads are symmetric for SAME
    h_pads = pads[0] + pads[2] if len(pads) >= 4 else 0
    w_pads = pads[1] + pads[3] if len(pads) >= 4 else pads[0] + pads[1]
    if h_pads == kernel[0] - 1 and w_pads == kernel[1] - 1:
        return 0  # SAME
    return 1  # VALID


@_register("Conv")
def _conv(builder, node):
    attrs = _decode_conv_attrs(node)
    in_c = attrs["group"]
    out_c = attrs["group"]  # placeholder

    # Get inputs
    x_idx = builder._tensor_map[node.input[0]]
    # Ensure NHWC
    x_idx = builder.ensure_nhwc(x_idx)
    x_shape = builder._tensors[x_idx].shape  # NHWC

    # Weight: ONNX format (M, C/group, kH, kW) → TFLite format (O, H, W, I)
    w_name = node.input[1]
    w_data = builder.onnx_weights[w_name].copy()
    in_c = w_data.shape[1] * attrs["group"]
    out_c = w_data.shape[0]
    w_data = w_data.transpose(0, 2, 3, 1)  # OIHW → OHWI
    w_idx = builder.register_weight(w_name + "_tfl", w_data)

    inputs = [x_idx, w_idx]
    if len(node.input) >= 3:
        b_name = node.input[2]
        b_data = builder.onnx_weights[b_name]
        b_idx = builder.register_weight(b_name + "_tfl", b_data)
        inputs.append(b_idx)
    else:
        # TFLite Conv requires bias; add zero bias
        zero_bias = np.zeros(out_c, dtype=np.float32)
        b_idx = builder.register_weight(f"{w_name}_zero_bias", zero_bias)
        inputs.append(b_idx)

    # Compute output shape (NHWC)
    p = attrs["pads"]
    s = attrs["strides"]
    k = attrs["kernel"]
    d = attrs["dilations"]
    h_out = (x_shape[1] + p[0] + p[2] - d[0] * (k[0] - 1) - 1) // s[0] + 1
    w_out = (x_shape[2] + p[1] + p[3] - d[1] * (k[1] - 1) - 1) // s[1] + 1
    out_shape = [x_shape[0], h_out, w_out, out_c]

    out = builder.register_tensor(node.output[0], out_shape, layout=Layout.Channel_Last)

    opt = Conv2DOptionsT()
    opt.strideH = s[0]
    opt.strideW = s[1]
    opt.dilationHFactor = d[0]
    opt.dilationWFactor = d[1]
    opt.padding = _conv_padding(p, s, k, attrs["auto_pad"])
    opt.fusedActivationFunction = 0  # NONE

    builder.add_op(Op.CONV_2D, inputs, [out], opt)
    builder.set_layout(out, Layout.Channel_Last)
    return [out]


@_register("ConvTranspose")
def _conv_transpose(builder, node):
    attrs = _decode_conv_attrs(node)

    x_idx = builder._tensor_map[node.input[0]]
    x_idx = builder.ensure_nhwc(x_idx)
    x_shape = builder._tensors[x_idx].shape

    w_name = node.input[1]
    w_data = builder.onnx_weights[w_name].copy()  # ONNX: (I, O, kH, kW)
    out_c = w_data.shape[1] * attrs["group"]
    in_c = w_data.shape[0]
    w_data = w_data.transpose(1, 2, 3, 0)  # IOHW → OHWI
    w_idx = builder.register_weight(w_name + "_tfl", w_data)

    inputs = [x_idx, w_idx]
    out_shape = [x_shape[0], x_shape[1] * attrs["strides"][0], x_shape[2] * attrs["strides"][1], out_c]

    s = attrs["strides"]
    opt = TransposeConvOptionsT()
    opt.strideH = s[0]
    opt.strideW = s[1]
    opt.padding = _conv_padding(attrs["pads"], s, attrs["kernel"], attrs["auto_pad"])

    out = builder.register_tensor(node.output[0], out_shape, layout=Layout.Channel_Last)
    builder.add_op(Op.TRANSPOSE_CONV, inputs, [out], opt)
    builder.set_layout(out, Layout.Channel_Last)
    return [out]


@_register("DepthwiseConv")
def _depthwise_conv(builder, node):
    # Not a standard ONNX op — PyTorch exports this as Conv with group=in_channels=out_channels
    # Redirect to Conv handler
    return _conv(builder, node)
