"""Deformation operators: Reshape, Transpose, Concat, Slice, Split, Gather, Tile, etc."""
import numpy as np
from tensorflow.lite.python.schema_py_generated import (
    BuiltinOperator as Op,
    ReshapeOptionsT,
    TransposeOptionsT,
    ConcatenationOptionsT,
    SqueezeOptionsT,
    ExpandDimsOptionsT,
    StridedSliceOptionsT,
    GatherOptionsT,
    TileOptionsT,
    SplitOptionsT,
    PadOptionsT,
    DepthToSpaceOptionsT,
)
from onnx2tflite.components.tflite_ir.builder import Layout
from onnx2tflite.components.tflite_ir.op_mapping import _register


def _get_attr(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            if attr.ints:
                return list(attr.ints)
            if attr.i or attr.i == 0:
                return attr.i
            if attr.f:
                return attr.f
    return default


@_register("Reshape")
def _reshape(builder, node):
    x_idx = builder._tensor_map[node.input[0]]
    shape_data = builder.onnx_weights.get(node.input[1])
    if shape_data is None:
        shape_idx = builder._tensor_map[node.input[1]]
    else:
        shape_idx = builder.register_weight(f"{node.output[0]}_shape", shape_data.astype(np.int32))

    out_shape = [int(s) for s in shape_data] if shape_data is not None else builder._tensors[x_idx].shape
    opt = ReshapeOptionsT()
    opt.newShape = out_shape
    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.RESHAPE, [x_idx, shape_idx] if shape_data is None else [x_idx], [out], opt)
    builder.set_layout(out, Layout.Channel_None if len(out_shape) <= 2 else Layout.Channel_First)
    return [out]


@_register("Transpose")
def _transpose(builder, node):
    x_idx = builder._tensor_map[node.input[0]]
    perm = _get_attr(node, "perm", [])
    in_shape = builder._tensors[x_idx].shape
    out_shape = [in_shape[p] for p in perm]

    perm_idx = builder.register_weight(f"{node.output[0]}_perm", np.array(perm, dtype=np.int32))
    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.TRANSPOSE, [x_idx, perm_idx], [out])
    builder.set_layout(out, Layout.Channel_First)
    return [out]


@_register("Concat")
def _concat(builder, node):
    axis = _get_attr(node, "axis", 0)
    inputs = [builder._tensor_map[n] for n in node.input if n in builder._tensor_map]
    # Compute output shape
    shapes = [builder._tensors[i].shape for i in inputs]
    out_shape = list(shapes[0])
    concat_dim = sum(s[axis] for s in shapes)
    out_shape[axis] = concat_dim

    opt = ConcatenationOptionsT()
    opt.axis = axis
    opt.fusedActivationFunction = 0
    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.CONCATENATION, inputs, [out], opt)
    builder.set_layout(out, builder.get_layout(inputs[0]))
    return [out]


@_register("Squeeze")
def _squeeze(builder, node):
    x_idx = builder._tensor_map[node.input[0]]
    axes = _get_attr(node, "axes", [1])
    if not isinstance(axes, list):
        axes = [int(axes)]
    in_shape = builder._tensors[x_idx].shape
    out_shape = [in_shape[i] for i in range(len(in_shape)) if i not in axes]

    opt = SqueezeOptionsT()
    opt.squeezeDims = axes
    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.SQUEEZE, [x_idx], [out], opt)
    builder.set_layout(out, Layout.Channel_None if len(out_shape) <= 2 else builder.get_layout(x_idx))
    return [out]


@_register("Unsqueeze")
def _unsqueeze(builder, node):
    x_idx = builder._tensor_map[node.input[0]]
    axes = _get_attr(node, "axes", [1])
    if not isinstance(axes, list):
        axes = [int(axes)]

    # Handle axes as input (opset 13+)
    if axes is None and len(node.input) > 1 and node.input[1] in builder.onnx_weights:
        axes = list(builder.onnx_weights[node.input[1]])

    opt = ExpandDimsOptionsT()
    # TFLite ExpandDims only handles single axis
    out_idx = x_idx
    out_shape = list(builder._tensors[x_idx].shape)
    for ax in sorted(axes):
        out_shape.insert(ax, 1)
        tmp = builder.register_tensor(f"{node.output[0]}_exp{ax}", out_shape)
        builder.add_op(Op.EXPAND_DIMS, [out_idx], [tmp], opt)
        out_idx = tmp

    # Rename final to output name
    builder._tensors[out_idx].name = node.output[0].encode()
    builder._tensor_map[node.output[0]] = out_idx
    builder.set_layout(out_idx, builder.get_layout(x_idx))
    return [out_idx]


@_register("Slice")
def _slice(builder, node):
    # ONNX Slice: data, starts, ends, axes[, steps]
    x_idx = builder._tensor_map[node.input[0]]
    shape = builder._tensors[x_idx].shape

    # Get starts/ends from inputs (opset 10+)
    starts = builder.onnx_weights.get(node.input[1], np.array([0], dtype=np.int32))
    ends = builder.onnx_weights.get(node.input[2], np.array(shape, dtype=np.int32))
    axes = builder.onnx_weights.get(node.input[3], np.array(list(range(len(shape))), dtype=np.int32)) if len(node.input) > 3 else np.array(list(range(len(shape))), dtype=np.int32)

    begin = [0] * len(shape)
    end = list(shape)
    strides = [1] * len(shape)
    for i, ax in enumerate(axes):
        begin[ax] = int(starts[i]) if i < len(starts) else 0
        end[ax] = int(ends[i]) if i < len(ends) else shape[ax]

    begin_mask = 0
    end_mask = 0
    out_shape = []
    for i in range(len(shape)):
        d = end[i] - begin[i]
        out_shape.append(max(1, d))

    opt = StridedSliceOptionsT()
    opt.begin = begin
    opt.end = end
    opt.strides = strides
    opt.beginMask = begin_mask
    opt.endMask = end_mask
    opt.ellipsisMask = 0
    opt.newAxisMask = 0
    opt.shrinkAxisMask = 0

    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.STRIDED_SLICE, [x_idx], [out], opt)
    builder.set_layout(out, builder.get_layout(x_idx))
    return [out]


@_register("Gather")
def _gather(builder, node):
    x_idx = builder._tensor_map[node.input[0]]
    indices = builder.onnx_weights.get(node.input[1])
    if indices is not None:
        idx = builder.register_weight(f"{node.output[0]}_idx", indices.astype(np.int32))
    else:
        idx = builder._tensor_map[node.input[1]]
    axis = _get_attr(node, "axis", 0)

    opt = GatherOptionsT()
    opt.axis = axis
    x_shape = builder._tensors[x_idx].shape
    idx_shape = builder._tensors[idx].shape
    out_shape = list(x_shape[:axis]) + list(idx_shape) + list(x_shape[axis + 1:])
    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.GATHER, [x_idx, idx], [out], opt)
    builder.set_layout(out, builder.get_layout(x_idx))
    return [out]


@_register("Tile")
def _tile(builder, node):
    x_idx = builder._tensor_map[node.input[0]]
    repeats = builder.onnx_weights.get(node.input[1])
    if repeats is None:
        raise ValueError("Tile requires repeats as initializer")
    repeat_idx = builder.register_weight(f"{node.output[0]}_rep", repeats.astype(np.int32))

    opt = TileOptionsT()
    out_shape = [builder._tensors[x_idx].shape[i] * int(repeats[i]) for i in range(len(repeats))]
    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.TILE, [x_idx, repeat_idx], [out], opt)
    builder.set_layout(out, builder.get_layout(x_idx))
    return [out]


@_register("Split")
def _split(builder, node):
    x_idx = builder._tensor_map[node.input[0]]
    axis = _get_attr(node, "axis", 0)
    num_splits = len(node.output)

    opt = SplitOptionsT()
    opt.numSplits = num_splits

    outputs = []
    x_shape = builder._tensors[x_idx].shape
    split_size = x_shape[axis] // num_splits
    for i, out_name in enumerate(node.output):
        out_shape = list(x_shape)
        out_shape[axis] = split_size
        out = builder.register_tensor(out_name, out_shape)
        outputs.append(out)

    builder.add_op(Op.SPLIT, [x_idx], outputs, opt)
    for o in outputs:
        builder.set_layout(o, builder.get_layout(x_idx))
    return outputs


@_register("Pad")
def _pad(builder, node):
    x_idx = builder._tensor_map[node.input[0]]
    pads_data = builder.onnx_weights.get(node.input[1]) if len(node.input) > 1 else None
    if pads_data is None:
        pads_data = np.array(_get_attr(node, "pads", [0, 0, 0, 0]), dtype=np.int32)

    x_shape = builder._tensors[x_idx].shape
    out_shape = list(x_shape)
    # pads format: [dim0_begin, dim1_begin, ..., dim0_end, dim1_end, ...]
    n = len(x_shape)
    for i in range(n):
        out_shape[i] = x_shape[i] + pads_data[i] + pads_data[i + n]

    opt = PadOptionsT()
    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.PAD, [x_idx], [out], opt)
    builder.set_layout(out, builder.get_layout(x_idx))
    return [out]


@_register("Flatten")
def _flatten(builder, node):
    x_idx = builder._tensor_map[node.input[0]]
    axis = _get_attr(node, "axis", 1)
    shape = builder._tensors[x_idx].shape
    # Flatten: first axis-1 dims kept, rest flattened
    flat_dim = 1
    for d in shape[axis:]:
        flat_dim *= d
    out_shape = list(shape[:axis]) + [flat_dim]

    opt = ReshapeOptionsT()
    opt.newShape = out_shape
    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.RESHAPE, [x_idx], [out], opt)
    builder.set_layout(out, Layout.Channel_None)
    return [out]


@_register("DepthToSpace")
def _depth_to_space(builder, node):
    x_idx = builder._tensor_map[node.input[0]]
    blocksize = _get_attr(node, "blocksize", 2)
    shape = builder._tensors[x_idx].shape
    out_shape = [shape[0], shape[1] // (blocksize * blocksize), shape[2] * blocksize, shape[3] * blocksize]

    opt = DepthToSpaceOptionsT()
    opt.blockSize = blocksize
    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.DEPTH_TO_SPACE, [x_idx], [out], opt)
    builder.set_layout(out, builder.get_layout(x_idx))
    return [out]


@_register("Expand")
def _expand(builder, node):
    # Broadcast: use Reshape + Tile pattern, or BROADCAST_TO
    from tensorflow.lite.python.schema_py_generated import BroadcastToOptionsT
    x_idx = builder._tensor_map[node.input[0]]
    shape_data = builder.onnx_weights.get(node.input[1])
    if shape_data is None:
        raise ValueError("Expand requires shape as initializer")
    shape_idx = builder.register_weight(f"{node.output[0]}_shape", shape_data.astype(np.int32))
    out_shape = [int(s) for s in shape_data]

    opt = BroadcastToOptionsT()
    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.BROADCAST_TO, [x_idx, shape_idx], [out], opt)
    builder.set_layout(out, builder.get_layout(x_idx))
    return [out]


@_register("GatherElements")
def _gather_elements(builder, node):
    # Not natively supported; mark as unsupported for now
    raise NotImplementedError("GatherElements requires decompose — not yet implemented")


@_register("ScatterND")
def _scatter_nd(builder, node):
    from tensorflow.lite.python.schema_py_generated import ScatterNdOptionsT
    x_idx = builder._tensor_map[node.input[0]]
    indices_data = builder.onnx_weights.get(node.input[1])
    updates_data = builder.onnx_weights.get(node.input[2])
    idx = builder.register_weight(f"{node.output[0]}_idx", indices_data.astype(np.int32))
    upd = builder.register_weight(f"{node.output[0]}_upd", updates_data)

    opt = ScatterNdOptionsT()
    out = builder.register_tensor(node.output[0], builder._tensors[x_idx].shape)
    builder.add_op(Op.SCATTER_ND, [x_idx, idx, upd], [out], opt)
    builder.set_layout(out, builder.get_layout(x_idx))
    return [out]


@_register("TopK")
def _topk(builder, node):
    from tensorflow.lite.python.schema_py_generated import TopKV2OptionsT
    x_idx = builder._tensor_map[node.input[0]]
    k_val = _get_attr(node, "K", 1) or builder.onnx_weights.get(node.input[1], np.array([1], dtype=np.int32))
    if isinstance(k_val, (list, np.ndarray)):
        k_val = int(k_val[0]) if len(k_val) > 0 else 1

    opt = TopKV2OptionsT()
    x_shape = builder._tensors[x_idx].shape
    out_shape = list(x_shape)
    out_shape[-1] = min(k_val, x_shape[-1])

    val_out = builder.register_tensor(f"{node.output[0]}_val", out_shape)
    idx_out = builder.register_tensor(f"{node.output[1]}_idx", out_shape, dtype=4)
    builder.add_op(Op.TOPK_V2, [x_idx], [val_out, idx_out], opt)
    return [val_out, idx_out]
