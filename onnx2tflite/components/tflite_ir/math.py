"""Math operators: Add, Sub, Mul, Div, Pow, MatMul, Reciprocal, Sqrt, etc."""
import numpy as np
from tensorflow.lite.python.schema_py_generated import BuiltinOperator as Op
from onnx2tflite.components.tflite_ir.builder import Layout
from onnx2tflite.components.tflite_ir.op_mapping import _register


def _get_inputs(builder, node):
    """Resolve inputs — tensor or weight."""
    result = []
    for name in node.input:
        if name in builder._tensor_map:
            result.append(builder._tensor_map[name])
        elif name in builder.onnx_weights:
            result.append(builder.register_weight(name, builder.onnx_weights[name]))
        else:
            raise KeyError(f"Input '{name}' not found in tensors or weights")
    return result


def _binary(builder, node, op_code, options_cls=None):
    inputs = _get_inputs(builder, node)
    out = builder.register_tensor(node.output[0], builder._tensors[inputs[0]].shape)
    opt = options_cls() if options_cls else None
    if options_cls:
        opt.fusedActivationFunction = 0  # NONE
    builder.add_op(op_code, inputs, [out], opt)
    builder.set_layout(out, builder.get_layout(inputs[0]))
    return [out]


@_register("Add")
def _add(builder, node):
    from tensorflow.lite.python.schema_py_generated import AddOptionsT
    return _binary(builder, node, Op.ADD, AddOptionsT)

@_register("Sub")
def _sub(builder, node):
    from tensorflow.lite.python.schema_py_generated import SubOptionsT
    return _binary(builder, node, Op.SUB, SubOptionsT)

@_register("Mul")
def _mul(builder, node):
    from tensorflow.lite.python.schema_py_generated import MulOptionsT
    return _binary(builder, node, Op.MUL, MulOptionsT)

@_register("Div")
def _div(builder, node):
    from tensorflow.lite.python.schema_py_generated import DivOptionsT
    return _binary(builder, node, Op.DIV, DivOptionsT)

@_register("Pow")
def _pow(builder, node):
    from tensorflow.lite.python.schema_py_generated import PowOptionsT
    return _binary(builder, node, Op.POW, PowOptionsT)

@_register("Maximum")
def _maximum(builder, node):
    from tensorflow.lite.python.schema_py_generated import MaximumMinimumOptionsT
    return _binary(builder, node, Op.MAXIMUM, MaximumMinimumOptionsT)

@_register("Minimum")
def _minimum(builder, node):
    from tensorflow.lite.python.schema_py_generated import MaximumMinimumOptionsT
    return _binary(builder, node, Op.MINIMUM, MaximumMinimumOptionsT)

@_register("Reciprocal")
def _reciprocal(builder, node):
    inp = builder._tensor_map[node.input[0]]
    one_name = f"{node.output[0]}_one"
    one_data = np.array([1.0], dtype=np.float32)
    one_idx = builder.register_weight(one_name, one_data)
    out = builder.register_tensor(node.output[0], builder._tensors[inp].shape)
    builder.add_op(Op.DIV, [one_idx, inp], [out])
    builder.set_layout(out, builder.get_layout(inp))
    return [out]

@_register("MatMul")
def _matmul(builder, node):
    from tensorflow.lite.python.schema_py_generated import BatchMatMulOptionsT
    inputs = _get_inputs(builder, node)
    opt = BatchMatMulOptionsT()
    opt.adjX = False
    opt.adjY = False
    # Determine output shape: A(M,K) @ B(K,N) -> (M,N)
    shape_a = builder._tensors[inputs[0]].shape
    shape_b = builder._tensors[inputs[1]].shape
    out_shape = list(shape_a[:-1]) + [shape_b[-1]] if len(shape_b) > 1 else list(shape_a[:-1])
    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.BATCH_MATMUL, inputs, [out], opt)
    builder.set_layout(out, Layout.Channel_None)
    return [out]

@_register("Gemm")
def _gemm(builder, node):
    # FULLY_CONNECTED expects filter (OutputUnits, InputUnits)
    # Re-register weight transposed
    from tensorflow.lite.python.schema_py_generated import FullyConnectedOptionsT
    w_data = builder.onnx_weights[node.input[1]]  # ONNX: (K, N) = (4, 8)
    w_t = w_data.T  # (8, 4)
    w_idx = builder.register_weight(f"{node.output[0]}_fc_w", w_t.astype(np.float32))
    inputs = [builder._tensor_map[node.input[0]], w_idx]
    if len(node.input) > 2:
        b_idx = builder.register_weight(f"{node.output[0]}_fc_b",
                                         builder.onnx_weights[node.input[2]].astype(np.float32))
        inputs.append(b_idx)
    out_features = w_t.shape[0]
    out_shape = [builder._tensors[inputs[0]].shape[0], out_features]
    opt = FullyConnectedOptionsT()
    opt.fusedActivationFunction = 0
    opt.weightsFormat = 0
    out = builder.register_tensor(node.output[0], out_shape)
    builder.add_op(Op.FULLY_CONNECTED, inputs, [out], opt)
    builder.set_layout(out, Layout.Channel_None)
    return [out]
