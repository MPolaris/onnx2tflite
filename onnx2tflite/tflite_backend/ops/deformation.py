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
from onnx2tflite.tflite_backend.builder import Layout
from onnx2tflite.tflite_backend.op_mapping import _register


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

    # TFLite ExpandDims needs axis as 2nd input
    out_idx = x_idx
    out_shape = list(builder._tensors[x_idx].shape)
    for ax in sorted(axes):
        axis_idx = builder.register_weight(f"{node.output[0]}_axis{ax}", np.array([ax], dtype=np.int32))
        out_shape.insert(ax, 1)
        tmp = builder.register_tensor(f"{node.output[0]}_exp{ax}", out_shape)
        builder.add_op(Op.EXPAND_DIMS, [out_idx, axis_idx], [tmp])
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

    # TFLite StridedSlice needs begin, end, strides as inputs
    begin_idx = builder.register_weight(f"{node.output[0]}_begin", np.array(begin, dtype=np.int32))
    end_idx = builder.register_weight(f"{node.output[0]}_end", np.array(end, dtype=np.int32))
    stride_idx = builder.register_weight(f"{node.output[0]}_strides", np.array(strides, dtype=np.int32))

    opt = StridedSliceOptionsT()
    opt.beginMask = begin_mask
    opt.endMask = end_mask
    opt.ellipsisMask = 0
    opt.newAxisMask = 0
    opt.shrinkAxisMask = 0

    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.STRIDED_SLICE, [x_idx, begin_idx, end_idx, stride_idx], [out], opt)
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
    # Read split sizes from attribute or input
    split_sizes = _get_attr(node, "split", None)
    if split_sizes is None and len(node.input) > 1:
        split_sizes = list(builder.onnx_weights.get(node.input[1], [2, 1]))
    if split_sizes is None:
        num_splits = len(node.output)
        split_sizes = [builder._tensors[x_idx].shape[axis] // num_splits] * num_splits

    # TFLite uses SPLIT_V (102) with [input, split_sizes, split_dim]
    sizes_idx = builder.register_weight(f"{node.output[0]}_sizes", np.array(split_sizes, dtype=np.int32))
    axis_idx = builder.register_weight(f"{node.output[0]}_axis", np.array([axis], dtype=np.int32))

    from tensorflow.lite.python.schema_py_generated import SplitVOptionsT
    opt = SplitVOptionsT()
    opt.numSplits = len(split_sizes)

    x_shape = builder._tensors[x_idx].shape
    outputs = []
    for i, out_name in enumerate(node.output):
        out_shape = list(x_shape)
        out_shape[axis] = split_sizes[i] if i < len(split_sizes) else split_sizes[-1]
        out = builder.register_tensor(out_name, out_shape)
        outputs.append(out)

    builder.add_op(Op.SPLIT_V, [x_idx, sizes_idx, axis_idx], outputs, opt)
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
        out_shape[i] = x_shape[i] + int(pads_data[i]) + int(pads_data[i + n])

    # TFLite Pad needs paddings as 2nd input: reshape to (N, 2) format
    # ONNX pads: [begin0, begin1, ..., end0, end1, ...]
    # TFLite expects: [[begin0, end0], [begin1, end1], ...]
    tfl_pads = np.array([[pads_data[i], pads_data[i + n]] for i in range(n)], dtype=np.int32)
    pads_idx = builder.register_weight(f"{node.output[0]}_pads", tfl_pads)
    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.PAD, [x_idx, pads_idx], [out])
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
    import numpy as np
    x_idx = builder._tensor_map[node.input[0]]
    x_idx = builder.ensure_nhwc(x_idx)
    # blocksize is a single int attribute
    blocksize = 2
    for attr in node.attribute:
        if attr.name == "blocksize":
            blocksize = attr.i
            break
    shape = builder._tensors[x_idx].shape  # NHWC: (N, H, W, C)
    out_shape = [shape[0], shape[1] * blocksize, shape[2] * blocksize, shape[3] // (blocksize * blocksize)]

    opt = DepthToSpaceOptionsT()
    opt.blockSize = blocksize
    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.DEPTH_TO_SPACE, [x_idx], [out], opt)
    builder.set_layout(out, Layout.Channel_Last)
    return [out]


@_register("Expand")
def _expand(builder, node):
    # Use Reshape + Tile (BROADCAST_TO op 130 not in older TFLite schema)
    from tensorflow.lite.python.schema_py_generated import ReshapeOptionsT, TileOptionsT
    x_idx = builder._tensor_map[node.input[0]]
    shape_data = builder.onnx_weights.get(node.input[1])
    if shape_data is None:
        raise ValueError("Expand requires shape as initializer")
    out_shape = [int(s) for s in shape_data]
    x_shape = builder._tensors[x_idx].shape

    # Compute repeats for Tile
    repeats = []
    offset = len(out_shape) - len(x_shape)
    for i in range(len(out_shape)):
        if i < offset:
            repeats.append(out_shape[i])
        elif i - offset < len(x_shape) and x_shape[i - offset] == out_shape[i]:
            repeats.append(1)
        else:
            repeats.append(out_shape[i])

    # Reshape x to match ndim
    if len(x_shape) < len(out_shape):
        new_shape = [1] * offset + list(x_shape)
        reshape_out = builder.register_tensor(f"{node.output[0]}_r", new_shape)
        opt_r = ReshapeOptionsT(); opt_r.newShape = new_shape
        builder.add_op(Op.RESHAPE, [x_idx], [reshape_out], opt_r)
        x_idx = reshape_out

    rep_idx = builder.register_weight(f"{node.output[0]}_repeats", np.array(repeats, dtype=np.int32))
    opt = TileOptionsT()
    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.TILE, [x_idx, rep_idx], [out], opt)
    builder.set_layout(out, builder.get_layout(x_idx))
    return [out]


@_register("GatherElements")
def _gather_elements(builder, node):
    """ONNX GatherElements via GATHER_ND: build coordinate tensor then gather.
    For each output position, construct (d0, ..., d_axis=indices, ..., dN) coordinate.
    """
    data_idx = builder._tensor_map[node.input[0]]
    data_shape = list(builder._tensors[data_idx].shape)
    ndim = len(data_shape)
    axis = _get_attr(node, "axis", 1)
    if axis < 0:
        axis += ndim

    indices_data = builder.onnx_weights.get(node.input[1])
    if indices_data is None:
        raise NotImplementedError("GatherElements with dynamic indices not yet supported")
    indices_idx = builder.register_weight(f"{node.output[0]}_idx", indices_data.astype(np.int32))
    indices_shape = list(indices_data.shape)
    out_shape = indices_shape

    # Build coordinate tensors for each dimension
    coord_parts = []
    for d in range(ndim):
        if d == axis:
            coord_parts.append(indices_idx)
        else:
            # Create a constant mesh for dim d with shape matching indices_shape
            # Values are 0, 1, 2, ..., data_shape[d]-1, broadcast to indices_shape[d]
            mesh_vals = np.arange(data_shape[d], dtype=np.int32)
            # Reshape to broadcast correctly: size-1 everywhere except at dim d
            mesh_shape = [1] * ndim
            mesh_shape[d] = data_shape[d]
            mesh = mesh_vals.reshape(mesh_shape)
            # Broadcast to indices_shape
            broadcast_shape = [1] * ndim
            for i in range(ndim):
                if i == d:
                    broadcast_shape[i] = indices_shape[i]
                elif indices_shape[i] == data_shape[i]:
                    broadcast_shape[i] = indices_shape[i]
                else:
                    broadcast_shape[i] = 1
            # Actually, ONNX broadcasting: mesh shape must broadcast to indices_shape
            # Mesh already has 1s except at dim d. indices_shape at dim d determines the size.
            # Use TILE to expand mesh to match indices_shape
            repeats = [indices_shape[i] // mesh_shape[i] for i in range(ndim)]
            mesh_idx = builder.register_weight(f"{node.output[0]}_mesh{d}", mesh.astype(np.int32))
            if any(r > 1 for r in repeats):
                rep_idx = builder.register_weight(f"{node.output[0]}_rep{d}", np.array(repeats, dtype=np.int32))
                tiled = builder.register_tensor(f"{node.output[0]}_tile{d}", list(indices_shape), dtype=2)
                builder.add_op(Op.TILE, [mesh_idx, rep_idx], [tiled])
                coord_parts.append(tiled)
            else:
                # Need to reshape if shapes differ
                if mesh_shape != indices_shape:
                    reshaped = builder.register_tensor(f"{node.output[0]}_rsh{d}", list(indices_shape), dtype=2)
                    from tensorflow.lite.python.schema_py_generated import ReshapeOptionsT
                    opt = ReshapeOptionsT(); opt.newShape = list(indices_shape)
                    builder.add_op(Op.RESHAPE, [mesh_idx], [reshaped], opt)
                    coord_parts.append(reshaped)
                else:
                    coord_parts.append(mesh_idx)

    # Stack all coordinate parts into (..., ndim) tensor
    # Need to CONCATENATE along new axis — reshape each to (..., 1) then concat
    expanded = []
    for i, cp in enumerate(coord_parts):
        exp_shape = list(indices_shape) + [1]
        exp = builder.register_tensor(f"{node.output[0]}_exp{i}", exp_shape, dtype=2)
        from tensorflow.lite.python.schema_py_generated import ReshapeOptionsT
        opt = ReshapeOptionsT(); opt.newShape = exp_shape
        builder.add_op(Op.RESHAPE, [cp], [exp], opt)
        expanded.append(exp)

    coords_shape = list(indices_shape) + [ndim]
    coords = builder.register_tensor(f"{node.output[0]}_coords", coords_shape, dtype=2)
    from tensorflow.lite.python.schema_py_generated import ConcatenationOptionsT
    cat_opt = ConcatenationOptionsT()
    cat_opt.axis = ndim  # concatenate along the new last axis
    cat_opt.fusedActivationFunction = 0
    builder.add_op(Op.CONCATENATION, expanded, [coords], cat_opt)

    # GATHER_ND(data, coords) → output
    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.GATHER_ND, [data_idx, coords], [out])
    builder.set_layout(out, builder.get_layout(data_idx))
    return [out]


@_register("ScatterND")
def _scatter_nd(builder, node):
    # In-place update: result = data + SCATTER_ND(indices, updates - GATHER_ND(data, indices), shape)
    data_idx = builder._tensor_map[node.input[0]]
    data_shape = list(builder._tensors[data_idx].shape)
    indices_data = builder.onnx_weights.get(node.input[1])
    updates_data = builder.onnx_weights.get(node.input[2])

    idx = builder.register_weight(f"{node.output[0]}_idx", indices_data.astype(np.int32))
    upd = builder.register_weight(f"{node.output[0]}_upd", updates_data.astype(np.float32))
    shape_idx = builder.register_weight(f"{node.output[0]}_shape", np.array(data_shape, dtype=np.int32))

    # 1. old_values = GATHER_ND(data, indices)
    old_vals = builder.register_tensor(f"{node.output[0]}_old", list(updates_data.shape))
    builder.add_op(Op.GATHER_ND, [data_idx, idx], [old_vals])

    # 2. diff = SUB(updates, old_values)
    diff = builder.register_tensor(f"{node.output[0]}_diff", list(updates_data.shape))
    builder.add_op(Op.SUB, [upd, old_vals], [diff])

    # 3. scattered_diff = SCATTER_ND(indices, diff, output_shape)
    scattered_diff = builder.register_tensor(f"{node.output[0]}_sdiff", data_shape)
    builder.add_op(Op.SCATTER_ND, [idx, diff, shape_idx], [scattered_diff])

    # 4. result = ADD(data, scattered_diff)
    out = builder.register_tensor(node.output[0], data_shape)
    builder.add_op(Op.ADD, [data_idx, scattered_diff], [out])
    builder.set_layout(out, builder.get_layout(data_idx))
    return [out]


@_register("TopK")
def _topk(builder, node):
    from tensorflow.lite.python.schema_py_generated import TopKV2OptionsT
    x_idx = builder._tensor_map[node.input[0]]
    k_val = _get_attr(node, "K", 1) or builder.onnx_weights.get(node.input[1], np.array([1], dtype=np.int32))
    if isinstance(k_val, (list, np.ndarray)):
        k_val = int(k_val[0]) if len(k_val) > 0 else 1

    # TFLite TopKV2 needs k as 2nd input (scalar int32)
    k_idx = builder.register_weight(f"{node.output[0]}_k", np.array(k_val, dtype=np.int32))
    x_shape = builder._tensors[x_idx].shape
    out_shape = list(x_shape)
    out_shape[-1] = min(k_val, x_shape[-1])

    val_out = builder.register_tensor(node.output[0], out_shape)
    idx_out = builder.register_tensor(node.output[1], out_shape, dtype=4)
    builder.add_op(Op.TOPK_V2, [x_idx, k_idx], [val_out, idx_out])
    return [val_out, idx_out]
