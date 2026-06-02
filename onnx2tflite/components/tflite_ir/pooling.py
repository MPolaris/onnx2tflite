"""Pooling operators: MaxPool, AveragePool, GlobalMaxPool, GlobalAveragePool."""
from tensorflow.lite.python.schema_py_generated import (
    BuiltinOperator as Op,
    Pool2DOptionsT,
    ReducerOptionsT,
)
from onnx2tflite.components.tflite_ir.builder import Layout
from onnx2tflite.components.tflite_ir.op_mapping import _register


def _pool_attrs(node):
    attrs = {}
    for attr in node.attribute:
        if attr.name == "kernel_shape":
            attrs["kernel"] = list(attr.ints)
        elif attr.name == "strides":
            attrs["strides"] = list(attr.ints)
        elif attr.name == "pads":
            attrs["pads"] = list(attr.ints)
        elif attr.name == "ceil_mode":
            attrs["ceil_mode"] = attr.i
    attrs.setdefault("kernel", [2, 2])
    attrs.setdefault("strides", [1, 1])
    attrs.setdefault("pads", [0, 0, 0, 0])
    attrs.setdefault("ceil_mode", 0)
    return attrs


def _pool_output_shape(x_shape, kernel, strides, pads):
    """Compute NHWC output shape for pooling."""
    h = (x_shape[1] + pads[0] + pads[2] - kernel[0]) // strides[0] + 1
    w = (x_shape[2] + pads[1] + pads[3] - kernel[1]) // strides[1] + 1
    return [x_shape[0], h, w, x_shape[3]]


@_register("MaxPool")
def _max_pool(builder, node):
    attrs = _pool_attrs(node)
    x_idx = builder._tensor_map[node.input[0]]
    x_idx = builder.ensure_nhwc(x_idx)
    x_shape = builder._tensors[x_idx].shape
    out_shape = _pool_output_shape(x_shape, attrs["kernel"], attrs["strides"], attrs["pads"])

    opt = Pool2DOptionsT()
    opt.filterHeight = attrs["kernel"][0]
    opt.filterWidth = attrs["kernel"][1]
    opt.strideH = attrs["strides"][0]
    opt.strideW = attrs["strides"][1]
    opt.padding = 1 if sum(attrs["pads"]) == 0 else 0
    opt.fusedActivationFunction = 0

    out = builder.register_tensor(node.output[0], out_shape, layout=Layout.Channel_Last)
    builder.add_op(Op.MAX_POOL_2D, [x_idx], [out], opt)
    builder.set_layout(out, Layout.Channel_Last)
    return [out]


@_register("AveragePool")
def _avg_pool(builder, node):
    attrs = _pool_attrs(node)
    x_idx = builder._tensor_map[node.input[0]]
    x_idx = builder.ensure_nhwc(x_idx)
    x_shape = builder._tensors[x_idx].shape
    out_shape = _pool_output_shape(x_shape, attrs["kernel"], attrs["strides"], attrs["pads"])

    opt = Pool2DOptionsT()
    opt.filterHeight = attrs["kernel"][0]
    opt.filterWidth = attrs["kernel"][1]
    opt.strideH = attrs["strides"][0]
    opt.strideW = attrs["strides"][1]
    opt.padding = 1 if sum(attrs["pads"]) == 0 else 0
    opt.fusedActivationFunction = 0

    out = builder.register_tensor(node.output[0], out_shape, layout=Layout.Channel_Last)
    builder.add_op(Op.AVERAGE_POOL_2D, [x_idx], [out], opt)
    builder.set_layout(out, Layout.Channel_Last)
    return [out]


@_register("GlobalMaxPool")
def _global_max_pool(builder, node):
    import numpy as np
    x_idx = builder._tensor_map[node.input[0]]
    x_idx = builder.ensure_nhwc(x_idx)
    shape = builder._tensors[x_idx].shape
    out_shape = [shape[0], 1, 1, shape[3]]

    # REDUCE_MAX needs axes as second input
    axes_idx = builder.register_weight(f"{node.output[0]}_axes", np.array([1, 2], dtype=np.int32))
    opt = ReducerOptionsT()
    opt.keepDims = True
    out = builder.register_tensor(node.output[0], out_shape, layout=Layout.Channel_Last)
    builder.add_op(Op.REDUCE_MAX, [x_idx, axes_idx], [out], opt)
    builder.set_layout(out, Layout.Channel_Last)
    return [out]


@_register("GlobalAveragePool")
def _global_avg_pool(builder, node):
    import numpy as np
    x_idx = builder._tensor_map[node.input[0]]
    x_idx = builder.ensure_nhwc(x_idx)
    shape = builder._tensors[x_idx].shape
    out_shape = [shape[0], 1, 1, shape[3]]

    axes_idx = builder.register_weight(f"{node.output[0]}_axes", np.array([1, 2], dtype=np.int32))
    opt = ReducerOptionsT()
    opt.keepDims = True
    out = builder.register_tensor(node.output[0], out_shape, layout=Layout.Channel_Last)
    builder.add_op(Op.MEAN, [x_idx, axes_idx], [out], opt)
    builder.set_layout(out, Layout.Channel_Last)
    return [out]
