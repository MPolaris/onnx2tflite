"""Reduction operators: ReduceSum/Mean/Max/Min, ArgMax/ArgMin."""
from tensorflow.lite.python.schema_py_generated import (
    BuiltinOperator as Op,
    ReducerOptionsT,
    ArgMaxOptionsT,
    ArgMinOptionsT,
)
from onnx2tflite.tflite_backend.builder import Layout
from onnx2tflite.tflite_backend.op_mapping import _register


def _reduce_attrs(node):
    attrs = {}
    for attr in node.attribute:
        if attr.name == "axes":
            attrs["axes"] = list(attr.ints)
        elif attr.name == "keepdims":
            attrs["keepdims"] = attr.i == 1
    attrs.setdefault("axes", [-1])
    attrs.setdefault("keepdims", True)
    return attrs


@_register("ReduceSum")
def _reduce_sum(builder, node):
    import numpy as np
    attrs = _reduce_attrs(node)
    x_idx = builder._tensor_map[node.input[0]]
    axes_idx = builder.register_weight(f"{node.output[0]}_axes", np.array(attrs["axes"], dtype=np.int32))
    opt = ReducerOptionsT()
    opt.keepDims = attrs["keepdims"]
    out = builder.register_tensor(node.output[0], builder._tensors[x_idx].shape)
    builder.add_op(Op.SUM, [x_idx, axes_idx], [out], opt)
    builder.set_layout(out, builder.get_layout(x_idx))
    return [out]


@_register("ReduceMean")
def _reduce_mean(builder, node):
    import numpy as np
    attrs = _reduce_attrs(node)
    x_idx = builder._tensor_map[node.input[0]]
    axes_idx = builder.register_weight(f"{node.output[0]}_axes", np.array(attrs["axes"], dtype=np.int32))
    opt = ReducerOptionsT()
    opt.keepDims = attrs["keepdims"]
    out = builder.register_tensor(node.output[0], builder._tensors[x_idx].shape)
    builder.add_op(Op.MEAN, [x_idx, axes_idx], [out], opt)
    builder.set_layout(out, builder.get_layout(x_idx))
    return [out]


@_register("ReduceMax")
def _reduce_max(builder, node):
    import numpy as np
    attrs = _reduce_attrs(node)
    x_idx = builder._tensor_map[node.input[0]]
    axes_idx = builder.register_weight(f"{node.output[0]}_axes", np.array(attrs["axes"], dtype=np.int32))
    opt = ReducerOptionsT()
    opt.keepDims = attrs["keepdims"]
    out = builder.register_tensor(node.output[0], builder._tensors[x_idx].shape)
    builder.add_op(Op.REDUCE_MAX, [x_idx, axes_idx], [out], opt)
    builder.set_layout(out, builder.get_layout(x_idx))
    return [out]


@_register("ReduceMin")
def _reduce_min(builder, node):
    import numpy as np
    attrs = _reduce_attrs(node)
    x_idx = builder._tensor_map[node.input[0]]
    axes_idx = builder.register_weight(f"{node.output[0]}_axes", np.array(attrs["axes"], dtype=np.int32))
    opt = ReducerOptionsT()
    opt.keepDims = attrs["keepdims"]
    out = builder.register_tensor(node.output[0], builder._tensors[x_idx].shape)
    builder.add_op(Op.REDUCE_MIN, [x_idx, axes_idx], [out], opt)
    builder.set_layout(out, builder.get_layout(x_idx))
    return [out]


@_register("ArgMax")
def _argmax(builder, node):
    import numpy as np
    attrs = _reduce_attrs(node)
    x_idx = builder._tensor_map[node.input[0]]
    axis = attrs["axes"][0] if isinstance(attrs["axes"], list) else attrs["axes"]
    axis_idx = builder.register_weight(f"{node.output[0]}_axis", np.array([axis], dtype=np.int32))
    opt = ArgMaxOptionsT()
    opt.outputType = 4  # INT64
    out = builder.register_tensor(node.output[0], builder._tensors[x_idx].shape, dtype=4)
    builder.add_op(Op.ARG_MAX, [x_idx, axis_idx], [out], opt)
    return [out]


@_register("ArgMin")
def _argmin(builder, node):
    import numpy as np
    attrs = _reduce_attrs(node)
    x_idx = builder._tensor_map[node.input[0]]
    axis = attrs["axes"][0] if isinstance(attrs["axes"], list) else attrs["axes"]
    axis_idx = builder.register_weight(f"{node.output[0]}_axis", np.array([axis], dtype=np.int32))
    opt = ArgMinOptionsT()
    opt.outputType = 4  # INT64
    out = builder.register_tensor(node.output[0], builder._tensors[x_idx].shape, dtype=4)
    builder.add_op(Op.ARG_MIN, [x_idx, axis_idx], [out], opt)
    return [out]
