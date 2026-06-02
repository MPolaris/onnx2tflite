"""
Direct IR tests: ONNX runtime vs TFLite direct IR builder.
Tests the core ops that have 1:1 TFLite builtin mappings.
"""
import numpy as np
import helper
from onnx2tflite.converter import onnx_converter


def _test_op(op_type, inputs, initializers=None, attrs=None, opset=14):
    """Build ONNX graph, convert via direct IR, compare ONNX vs TFLite output."""
    import onnx
    from onnx import helper as H, numpy_helper

    input_names = [i[0] for i in inputs]
    init_names = [i[0] for i in (initializers or [])]
    output_names = [f"{op_type}_out"]
    node = H.make_node(op_type, input_names + init_names, output_names, **attrs if attrs else {})

    graph = H.make_graph([node], "test",
        [H.make_tensor_value_info(n, onnx.TensorProto.FLOAT, s) for n, s in inputs],
        [H.make_tensor_value_info(output_names[0], onnx.TensorProto.FLOAT, inputs[0][1])],
        initializer=[numpy_helper.from_array(v, name=n) for n, v in (initializers or [])])
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", opset)])

    path = f"unit_test/test_{op_type}_direct.onnx"
    onnx.save(model, path)

    result = onnx_converter(path, use_direct_ir=True, need_simplify=False)
    error = result["tflite_error"]
    assert error is not None, f"Direct IR conversion/check failed for {op_type}"
    assert error < 1e-3, f"{op_type} max error {error} >= 1e-3"
    return error


# ---- Activation ----

def test_relu():
    _test_op("Relu", [("X", (1, 3, 4, 4))])

def test_sigmoid():
    _test_op("Sigmoid", [("X", (1, 3, 4, 4))])

def test_tanh():
    _test_op("Tanh", [("X", (1, 3, 4, 4))])

def test_leaky_relu():
    _test_op("LeakyRelu", [("X", (1, 3, 4, 4))], attrs={"alpha": 0.01})

def test_elu():
    _test_op("Elu", [("X", (1, 3, 4, 4))])

def test_hard_swish():
    _test_op("HardSwish", [("X", (1, 3, 4, 4))])

def test_softmax():
    _test_op("Softmax", [("input", (1, 3, 4, 4))], attrs={"axis": -1})


# ---- Math ----

def test_add():
    _test_op("Add", [("A", (1, 3, 4, 4)), ("B", (1, 3, 4, 4))])

def test_sub():
    _test_op("Sub", [("A", (1, 3, 4, 4)), ("B", (1, 3, 4, 4))])

def test_mul():
    _test_op("Mul", [("A", (1, 3, 4, 4)), ("B", (1, 3, 4, 4))])

def test_div():
    _test_op("Div", [("A", (1, 3, 4, 4)), ("B", (1, 3, 4, 4))])

def test_pow():
    _test_op("Pow", [("X", (1, 3, 4, 4))], initializers=[("Y", np.array([2.0], dtype=np.float32))])


# ---- Conv ----

def test_conv2d():
    in_c, out_c = 3, 6
    w = np.random.randn(out_c, in_c, 3, 3).astype(np.float32)
    b = np.random.randn(out_c).astype(np.float32)
    _test_op("Conv", [("X", (1, in_c, 8, 8))],
             initializers=[("W", w), ("B", b)],
             attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1]})

def test_conv2d_strided():
    in_c, out_c = 3, 6
    w = np.random.randn(out_c, in_c, 3, 3).astype(np.float32)
    _test_op("Conv", [("X", (1, in_c, 16, 16))],
             initializers=[("W", w)],
             attrs={"kernel_shape": [3, 3], "strides": [2, 2], "pads": [0, 0, 0, 0]})


# ---- Pool ----

def test_max_pool():
    _test_op("MaxPool", [("X", (1, 3, 8, 8))],
             attrs={"kernel_shape": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]})

def test_average_pool():
    _test_op("AveragePool", [("X", (1, 3, 8, 8))],
             attrs={"kernel_shape": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]})

def test_global_average_pool():
    _test_op("GlobalAveragePool", [("X", (1, 3, 4, 4))])

def test_global_max_pool():
    _test_op("GlobalMaxPool", [("X", (1, 3, 4, 4))])


# ---- Deformation ----

def test_reshape():
    _test_op("Reshape", [("data", (1, 48))],
             initializers=[("shape", np.array([1, 3, 4, 4], dtype=np.int64))])

def test_transpose():
    _test_op("Transpose", [("data", (1, 3, 4, 4))],
             attrs={"perm": [0, 2, 3, 1]})

def test_concat():
    import onnx; from onnx import helper as H
    node = H.make_node("Concat", ["A", "B", "C"], ["Y"], axis=1)
    graph = H.make_graph([node], "test",
        [H.make_tensor_value_info("A", onnx.TensorProto.FLOAT, (1, 2, 4, 4)),
         H.make_tensor_value_info("B", onnx.TensorProto.FLOAT, (1, 3, 4, 4)),
         H.make_tensor_value_info("C", onnx.TensorProto.FLOAT, (1, 1, 4, 4))],
        [H.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, (1, 6, 4, 4))])
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", 14)])
    onnx.save(model, "unit_test/test_concat_direct.onnx")
    r = onnx_converter("unit_test/test_concat_direct.onnx", use_direct_ir=True, need_simplify=False)
    assert r["tflite_error"] is not None and r["tflite_error"] < 1e-3


# ---- Decompose ----

def test_celu():
    _test_op("Celu", [("X", (1, 3, 4, 4))], attrs={"alpha": 1.0})

def test_hard_sigmoid():
    _test_op("HardSigmoid", [("X", (1, 3, 4, 4))], attrs={"alpha": 0.2, "beta": 0.5})

def test_mish():
    # Mish is a custom op — onnxruntime can't execute it
    # The direct IR build still works but error checking fails
    import pytest
    pytest.skip("Mish is a custom ONNX op — onnxruntime cannot execute it")

def test_softplus():
    _test_op("Softplus", [("X", (1, 3, 4, 4))])

def test_softsign():
    _test_op("Softsign", [("X", (1, 3, 4, 4))])

def test_clip():
    # Opset 11+: min/max are optional INPUTS, not attributes
    _test_op("Clip", [("input", (1, 3, 4, 4))],
             initializers=[("min_val", np.array(0.0, dtype=np.float32)),
                           ("max_val", np.array(6.0, dtype=np.float32))])
