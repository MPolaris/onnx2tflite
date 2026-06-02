"""
ONNX op → TFLite BuiltinOperator dispatch table.

Each handler function takes (builder, node) and returns the list of output tensor indices.
"""
from tensorflow.lite.python.schema_py_generated import BuiltinOperator as Op

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
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    builder.add_op(Op.RESHAPE, [inp], [out])  # use reshape as passthrough
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("Dropout")
def _dropout(builder, node):
    inp = builder._tensor_map[node.input[0]]
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    builder.add_op(Op.RESHAPE, [inp], [out])  # passthrough
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

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
