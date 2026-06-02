"""Convolution operators: Conv, ConvTranspose, DepthwiseConv."""
import numpy as np
from tensorflow.lite.python.schema_py_generated import (
    BuiltinOperator as Op,
    Conv2DOptionsT,
    DepthwiseConv2DOptionsT,
    TransposeConvOptionsT,
    ConcatenationOptionsT,
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

    x_idx = builder._tensor_map[node.input[0]]
    x_idx = builder.ensure_nhwc(x_idx)
    x_shape = builder._tensors[x_idx].shape  # NHWC
    x_c = x_shape[3]  # input channels in NHWC

    w_name = node.input[1]
    w_data = builder.onnx_weights[w_name].copy()  # ONNX: (M, C/group, kH, kW)
    out_c = w_data.shape[0]
    in_c = w_data.shape[1] * attrs["group"]
    group = attrs["group"]
    p, s, k, d = attrs["pads"], attrs["strides"], attrs["kernel"], attrs["dilations"]
    h_out = (x_shape[1] + p[0] + p[2] - d[0] * (k[0] - 1) - 1) // s[0] + 1
    w_out = (x_shape[2] + p[1] + p[3] - d[1] * (k[1] - 1) - 1) // s[1] + 1
    out_shape = [x_shape[0], h_out, w_out, out_c]

    # Bias
    if len(node.input) >= 3:
        b_data = builder.onnx_weights[node.input[2]]
        b_idx = builder.register_weight(f"{node.output[0]}_bias", b_data.astype(np.float32))
    else:
        b_idx = builder.register_weight(f"{node.output[0]}_zbias", np.zeros(out_c, dtype=np.float32))

    if group == 1:
        # ---- Standard convolution ----
        # ONNX (M, C, H, W) → TFLite OHWI (O, H, W, I)
        w_tfl = w_data.transpose(0, 2, 3, 1)
        w_idx = builder.register_weight(f"{w_name}_tfl", w_tfl)
        opt = Conv2DOptionsT()
        opt.strideH, opt.strideW = s[0], s[1]
        opt.dilationHFactor, opt.dilationWFactor = d[0], d[1]
        opt.padding = _conv_padding(p, s, k, attrs["auto_pad"])
        opt.fusedActivationFunction = 0
        out = builder.register_tensor(node.output[0], out_shape, layout=Layout.Channel_Last)
        builder.add_op(Op.CONV_2D, [x_idx, w_idx, b_idx], [out], opt)
        builder.set_layout(out, Layout.Channel_Last)
        return [out]

    elif group == in_c and group == out_c:
        # ---- Depthwise convolution ----
        # ONNX (C, 1, H, W) → TFLite depthwise (1, H, W, C)
        w_tfl = w_data[:, 0, :, :]  # (C, H, W)
        w_tfl = w_tfl.transpose(1, 2, 0)  # (H, W, C)
        w_tfl = w_tfl[np.newaxis, ...]  # (1, H, W, C)
        w_idx = builder.register_weight(f"{w_name}_dw", w_tfl)
        opt = DepthwiseConv2DOptionsT()
        opt.strideH, opt.strideW = s[0], s[1]
        opt.dilationHFactor, opt.dilationWFactor = d[0], d[1]
        opt.depthMultiplier = 1
        opt.padding = _conv_padding(p, s, k, attrs["auto_pad"])
        opt.fusedActivationFunction = 0
        out = builder.register_tensor(node.output[0], out_shape, layout=Layout.Channel_Last)
        # BUG in TFLite 2.12: DEPTHWISE_CONV_2D with bias input expects [input, filter, bias]
        # But the schema expects [input, filter, bias] ... let me check
        builder.add_op(Op.DEPTHWISE_CONV_2D, [x_idx, w_idx, b_idx], [out], opt)
        builder.set_layout(out, Layout.Channel_Last)
        return [out]

    else:
        # ---- Grouped convolution via split ----
        # Split input along channel dim into `group` parts, conv each, then concat
        ch_per_group = in_c // group
        out_per_group = out_c // group

        # Split input: SPLIT_V into `group` parts along axis=-1 (channel)
        split_sizes = [ch_per_group] * group
        sizes_idx = builder.register_weight(f"{node.output[0]}_splitsz", np.array(split_sizes, dtype=np.int32))
        axis_idx = builder.register_weight(f"{node.output[0]}_splax", np.array([x_c - 1], dtype=np.int32))  # channel dim in NHWC

        split_outputs = []
        for g in range(group):
            g_shape = [x_shape[0], x_shape[1], x_shape[2], ch_per_group]
            g_out = builder.register_tensor(f"{node.output[0]}_split{g}", g_shape)
            split_outputs.append(g_out)

        from tensorflow.lite.python.schema_py_generated import SplitVOptionsT
        sopt = SplitVOptionsT()
        sopt.numSplits = group
        builder.add_op(Op.SPLIT_V, [x_idx, sizes_idx, axis_idx], split_outputs, sopt)

        # Conv each group
        conv_outputs = []
        for g in range(group):
            g_w = w_data[g*out_per_group:(g+1)*out_per_group, :, :, :]  # (out/g, ch/g, H, W)
            g_w_tfl = g_w.transpose(0, 2, 3, 1)  # OHWI
            g_w_idx = builder.register_weight(f"{w_name}_g{g}_tfl", g_w_tfl)

            # Bias for this group
            if len(node.input) >= 3:
                g_b_data = builder.onnx_weights[node.input[2]]
                if g_b_data.ndim > 0:
                    g_b = g_b_data[g*out_per_group:(g+1)*out_per_group]
                else:
                    g_b = g_b_data
                g_b_idx = builder.register_weight(f"{node.output[0]}_gbias{g}", g_b.astype(np.float32))
            else:
                g_b_idx = builder.register_weight(f"{node.output[0]}_gzbias{g}", np.zeros(out_per_group, dtype=np.float32))

            g_out_shape = [x_shape[0], h_out, w_out, out_per_group]
            g_conv_out = builder.register_tensor(f"{node.output[0]}_gconv{g}", g_out_shape)

            g_opt = Conv2DOptionsT()
            g_opt.strideH, g_opt.strideW = s[0], s[1]
            g_opt.dilationHFactor, g_opt.dilationWFactor = d[0], d[1]
            g_opt.padding = _conv_padding(p, s, k, attrs["auto_pad"])
            g_opt.fusedActivationFunction = 0
            builder.add_op(Op.CONV_2D, [split_outputs[g], g_w_idx, g_b_idx], [g_conv_out], g_opt)
            conv_outputs.append(g_conv_out)

        # Concat results along channel dim
        cat_opt = ConcatenationOptionsT()
        cat_opt.axis = x_c - 1  # channel dim in NHWC
        cat_opt.fusedActivationFunction = 0
        out = builder.register_tensor(node.output[0], out_shape, layout=Layout.Channel_Last)
        builder.add_op(Op.CONCATENATION, conv_outputs, [out], cat_opt)
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

    k = attrs["kernel"]
    s = attrs["strides"]
    p = attrs["pads"]
    # ConvTranspose output: H_out = (H_in - 1) * stride_h + kernel_h - 2*pad_top
    h_out = (x_shape[1] - 1) * s[0] + k[0] - p[0] * 2
    w_out = (x_shape[2] - 1) * s[1] + k[1] - p[1] * 2
    out_shape = [x_shape[0], h_out, w_out, out_c]

    # TFLite TransposeConv: inputs are [output_shape, weights, input] (no bias needed)
    os_idx = builder.register_weight(w_name + '_outshape',
        np.array(out_shape, dtype=np.int32))

    opt = TransposeConvOptionsT()
    opt.strideH = s[0]
    opt.strideW = s[1]
    opt.padding = _conv_padding(attrs["pads"], s, attrs["kernel"], attrs["auto_pad"])

    out = builder.register_tensor(node.output[0], out_shape, layout=Layout.Channel_Last)
    builder.add_op(Op.TRANSPOSE_CONV, [os_idx, w_idx, x_idx], [out], opt)
    builder.set_layout(out, Layout.Channel_Last)
    return [out]


@_register("DepthwiseConv")
def _depthwise_conv(builder, node):
    # Not a standard ONNX op — PyTorch exports this as Conv with group=in_channels=out_channels
    # Redirect to Conv handler
    return _conv(builder, node)
