"""
ONNX op → TFLite BuiltinOperator dispatch table.

Each handler function takes (builder, node) and returns the list of output tensor indices.
"""
import numpy as np
from tensorflow.lite.python.schema_py_generated import BuiltinOperator as Op
from onnx2tflite.tflite_backend.builder import Layout

# ONNX op name → handler function
_OP_MAP: dict[str, callable] = {}


def _register(op_name: str):
    """Decorator to register a handler for an ONNX op name."""
    def decorator(func):
        _OP_MAP[op_name] = func
        return func
    return decorator


def dispatch(builder, node) -> list[int]:
    """Look up and call the handler for a given ONNX node."""
    op_type = node.op_type
    handler = _OP_MAP.get(op_type)
    if handler is None:
        raise KeyError(f"TFLite IR: '{op_type}' not implemented in direct path")
    return handler(builder, node)


# ---- Direct 1:1 mappings ----

@_register("Relu")
def _relu(builder, node):
    inp = builder._tensor_map[node.input[0]]
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    builder.add_op(Op.RELU, [inp], [out])
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("Sigmoid")
def _sigmoid(builder, node):
    inp = builder._tensor_map[node.input[0]]
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    builder.add_op(Op.LOGISTIC, [inp], [out])
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("Tanh")
def _tanh(builder, node):
    inp = builder._tensor_map[node.input[0]]
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    builder.add_op(Op.TANH, [inp], [out])
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("LeakyRelu")
def _leaky_relu(builder, node):
    inp = builder._tensor_map[node.input[0]]
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    from tensorflow.lite.python.schema_py_generated import LeakyReluOptionsT
    opt = LeakyReluOptionsT()
    opt.alpha = node.attribute[0].f if node.attribute else 0.01
    builder.add_op(Op.LEAKY_RELU, [inp], [out], opt)
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("Elu")
def _elu(builder, node):
    inp = builder._tensor_map[node.input[0]]
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    builder.add_op(Op.ELU, [inp], [out])
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("PRelu")
def _prelu(builder, node):
    inp = builder._tensor_map[node.input[0]]
    slope_name = node.input[1]
    slope_idx = (builder._tensor_map.get(slope_name) or
                 builder.register_weight(slope_name, builder.onnx_weights[slope_name]))
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    builder.add_op(Op.PRELU, [inp, slope_idx], [out])
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("HardSwish")
def _hard_swish(builder, node):
    inp = builder._tensor_map[node.input[0]]
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    builder.add_op(Op.HARD_SWISH, [inp], [out])
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("Softmax")
def _softmax(builder, node):
    from tensorflow.lite.python.schema_py_generated import SoftmaxOptionsT
    inp = builder._tensor_map[node.input[0]]
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    opt = SoftmaxOptionsT()
    opt.beta = 1.0
    builder.add_op(Op.SOFTMAX, [inp], [out], opt)
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("Exp")
def _exp(builder, node):
    inp = builder._tensor_map[node.input[0]]
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    builder.add_op(Op.EXP, [inp], [out])
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("Log")
def _log(builder, node):
    inp = builder._tensor_map[node.input[0]]
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    builder.add_op(Op.LOG, [inp], [out])
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("Sqrt")
def _sqrt(builder, node):
    inp = builder._tensor_map[node.input[0]]
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    builder.add_op(Op.SQRT, [inp], [out])
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("Sin")
def _sin(builder, node):
    inp = builder._tensor_map[node.input[0]]
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    builder.add_op(Op.SIN, [inp], [out])
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("Cos")
def _cos(builder, node):
    inp = builder._tensor_map[node.input[0]]
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    builder.add_op(Op.COS, [inp], [out])
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("Sinh")
def _sinh(builder, node):
    # No native Sinh in TFLite — decompose: (e^x - e^(-x))/2
    inp = builder._tensor_map[node.input[0]]
    shape = builder._tensors[inp].shape
    layout = builder.get_layout(inp)
    neg_idx = builder.register_weight(f"{node.output[0]}_neg", np.array([-1.0], dtype=np.float32))
    two_idx = builder.register_weight(f"{node.output[0]}_two", np.array([2.0], dtype=np.float32))
    # e^x
    exp = builder.register_tensor(f"{node.output[0]}_exp", shape)
    builder.add_op(Op.EXP, [inp], [exp])
    # -x
    neg = builder.register_tensor(f"{node.output[0]}_negx", shape)
    builder.add_op(Op.MUL, [inp, neg_idx], [neg])
    # e^(-x)
    exp_neg = builder.register_tensor(f"{node.output[0]}_expn", shape)
    builder.add_op(Op.EXP, [neg], [exp_neg])
    # e^x - e^(-x)
    sub = builder.register_tensor(f"{node.output[0]}_sub", shape)
    builder.add_op(Op.SUB, [exp, exp_neg], [sub])
    # (e^x - e^(-x)) / 2
    out = builder.register_tensor(node.output[0], shape)
    builder.add_op(Op.DIV, [sub, two_idx], [out])
    builder.set_layout(out, layout)
    return [out]

@_register("Cosh")
def _cosh(builder, node):
    # No native Cosh — decompose: (e^x + e^(-x))/2
    inp = builder._tensor_map[node.input[0]]
    shape = builder._tensors[inp].shape
    layout = builder.get_layout(inp)
    neg_idx = builder.register_weight(f"{node.output[0]}_neg", np.array([-1.0], dtype=np.float32))
    two_idx = builder.register_weight(f"{node.output[0]}_two", np.array([2.0], dtype=np.float32))
    exp = builder.register_tensor(f"{node.output[0]}_exp", shape)
    builder.add_op(Op.EXP, [inp], [exp])
    neg = builder.register_tensor(f"{node.output[0]}_negx", shape)
    builder.add_op(Op.MUL, [inp, neg_idx], [neg])
    exp_neg = builder.register_tensor(f"{node.output[0]}_expn", shape)
    builder.add_op(Op.EXP, [neg], [exp_neg])
    add = builder.register_tensor(f"{node.output[0]}_add", shape)
    builder.add_op(Op.ADD, [exp, exp_neg], [add])
    out = builder.register_tensor(node.output[0], shape)
    builder.add_op(Op.DIV, [add, two_idx], [out])
    builder.set_layout(out, layout)
    return [out]

@_register("Tan")
def _tan(builder, node):
    # No native Tan — decompose: sin(x)/cos(x)
    inp = builder._tensor_map[node.input[0]]
    shape = builder._tensors[inp].shape
    layout = builder.get_layout(inp)
    sin_out = builder.register_tensor(f"{node.output[0]}_sin", shape)
    builder.add_op(Op.SIN, [inp], [sin_out])
    cos_out = builder.register_tensor(f"{node.output[0]}_cos", shape)
    builder.add_op(Op.COS, [inp], [cos_out])
    out = builder.register_tensor(node.output[0], shape)
    builder.add_op(Op.DIV, [sin_out, cos_out], [out])
    builder.set_layout(out, layout)
    return [out]

@_register("Resize")
def _resize(builder, node):
    from tensorflow.lite.python.schema_py_generated import ResizeNearestNeighborOptionsT, ResizeBilinearOptionsT
    x_idx = builder._tensor_map[node.input[0]]
    x_idx = builder.ensure_nhwc(x_idx)
    x_shape = builder._tensors[x_idx].shape

    mode = "nearest"
    for attr in node.attribute:
        if attr.name == "mode":
            mode = attr.s.decode() if hasattr(attr.s, 'decode') else str(attr.s)

    scale_data = builder.onnx_weights.get(node.input[2]) if len(node.input) > 2 else None
    if scale_data is not None and len(scale_data) >= 4:
        h_out = int(x_shape[1] * scale_data[2])
        w_out = int(x_shape[2] * scale_data[3])
    elif len(node.input) > 3 and node.input[3] in builder.onnx_weights:
        sizes = builder.onnx_weights[node.input[3]]
        if len(sizes) >= 4:
            h_out, w_out = int(sizes[2]), int(sizes[3])
        else:
            h_out, w_out = x_shape[1] * 2, x_shape[2] * 2
    else:
        h_out, w_out = x_shape[1] * 2, x_shape[2] * 2

    out_shape = [x_shape[0], h_out, w_out, x_shape[3]]
    out = builder.register_tensor(node.output[0], out_shape, layout=Layout.Channel_Last)

    # TFLite Resize needs size as 2nd input
    size_idx = builder.register_weight(f"{node.output[0]}_size", np.array([h_out, w_out], dtype=np.int32))

    if mode.lower() == "nearest":
        opt = ResizeNearestNeighborOptionsT()
        opt.alignCorners = False
        opt.halfPixelCenters = False
        builder.add_op(Op.RESIZE_NEAREST_NEIGHBOR, [x_idx, size_idx], [out], opt)
    else:
        opt = ResizeBilinearOptionsT()
        opt.alignCorners = False
        opt.halfPixelCenters = False
        builder.add_op(Op.RESIZE_BILINEAR, [x_idx, size_idx], [out], opt)
    builder.set_layout(out, Layout.Channel_Last)
    return [out]

@_register("Upsample")
def _upsample(builder, node):
    # Upsample is deprecated (use Resize), but handle it anyway
    return _resize(builder, node)

@_register("Abs")
def _abs(builder, node):
    inp = builder._tensor_map[node.input[0]]
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    builder.add_op(Op.ABS, [inp], [out])
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("Neg")
def _neg(builder, node):
    inp = builder._tensor_map[node.input[0]]
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    builder.add_op(Op.NEG, [inp], [out])
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("Identity")
def _identity(builder, node):
    inp = builder._tensor_map[node.input[0]]
    # Identity: alias the output to the same tensor (no-op in TFLite)
    builder._tensor_map[node.output[0]] = inp
    return [inp]

@_register("Dropout")
def _dropout(builder, node):
    # Dropout is no-op at inference time
    inp = builder._tensor_map[node.input[0]]
    builder._tensor_map[node.output[0]] = inp
    return [inp]

@_register("Cast")
def _cast(builder, node):
    from tensorflow.lite.python.schema_py_generated import CastOptionsT
    to_attr = None
    for attr in node.attribute:
        if attr.name == "to":
            to_attr = attr.i
            break
    inp = builder._tensor_map[node.input[0]]
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    opt = CastOptionsT()
    opt.inDataType = builder._tensors[inp].type
    opt.outDataType = to_attr if to_attr else 1
    builder.add_op(Op.CAST, [inp], [out], opt)
    return [out]
