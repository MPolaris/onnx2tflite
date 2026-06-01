"""
Unit tests for mathematics and reduction operators.

Reference: asserts/Operators.md for ONNX operator specs.
"""
import numpy as np
import helper


def test_add():
    helper.build_and_convert(
        nodes=[helper.make_node("Add", ["A", "B"], ["C"])],
        inputs=[helper.make_input("A", (1, 3, 4, 4)), helper.make_input("B", (1, 3, 4, 4))],
        outputs=[helper.make_output("C", (1, 3, 4, 4))],
        filename="test_add.onnx",
    )


def test_sub():
    helper.build_and_convert(
        nodes=[helper.make_node("Sub", ["A", "B"], ["C"])],
        inputs=[helper.make_input("A", (1, 3, 4, 4)), helper.make_input("B", (1, 3, 4, 4))],
        outputs=[helper.make_output("C", (1, 3, 4, 4))],
        filename="test_sub.onnx",
    )


def test_mul():
    helper.build_and_convert(
        nodes=[helper.make_node("Mul", ["A", "B"], ["C"])],
        inputs=[helper.make_input("A", (1, 3, 4, 4)), helper.make_input("B", (1, 3, 4, 4))],
        outputs=[helper.make_output("C", (1, 3, 4, 4))],
        filename="test_mul.onnx",
    )


def test_div():
    helper.build_and_convert(
        nodes=[helper.make_node("Div", ["A", "B"], ["C"])],
        inputs=[helper.make_input("A", (1, 3, 4, 4)), helper.make_input("B", (1, 3, 4, 4))],
        outputs=[helper.make_output("C", (1, 3, 4, 4))],
        filename="test_div.onnx",
    )


def test_mod():
    # fmod=1 required for floating point inputs per ONNX spec
    w = np.array([2.0], dtype=np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("Mod", ["A", "mod_val"], ["C"], fmod=1)],
        inputs=[helper.make_input("A", (1, 3, 4, 4))],
        outputs=[helper.make_output("C", (1, 3, 4, 4))],
        initializers=[helper.make_weight("mod_val", w)],
        filename="test_mod.onnx",
    )


def test_pow():
    exponent = np.array([2.0], dtype=np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("Pow", ["X", "Y"], ["Z"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Z", (1, 3, 4, 4))],
        initializers=[helper.make_weight("Y", exponent)],
        filename="test_pow.onnx",
    )


def test_reciprocal():
    helper.build_and_convert(
        nodes=[helper.make_node("Reciprocal", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_reciprocal.onnx",
    )


def test_sqrt():
    helper.build_and_convert(
        nodes=[helper.make_node("Sqrt", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_sqrt.onnx",
    )


def test_exp():
    helper.build_and_convert(
        nodes=[helper.make_node("Exp", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_exp.onnx",
    )


def test_log():
    helper.build_and_convert(
        nodes=[helper.make_node("Log", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_log.onnx",
    )


def test_erf():
    helper.build_and_convert(
        nodes=[helper.make_node("Erf", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_erf.onnx",
    )


def test_reduce_sum():
    # Opset 13+: axes is an INPUT tensor, not attribute
    axes_val = np.array([2, 3], dtype=np.int64)
    helper.build_and_convert(
        nodes=[helper.make_node("ReduceSum", ["data", "axes"], ["reduced"], keepdims=1)],
        inputs=[helper.make_input("data", (1, 3, 4, 4))],
        outputs=[helper.make_output("reduced", (1, 3, 1, 1))],
        initializers=[helper.make_weight("axes", axes_val)],
        filename="test_reduce_sum.onnx",
    )


def test_reduce_mean():
    helper.build_and_convert(
        nodes=[helper.make_node("ReduceMean", ["data"], ["reduced"], axes=[2, 3], keepdims=1)],
        inputs=[helper.make_input("data", (1, 3, 4, 4))],
        outputs=[helper.make_output("reduced", (1, 3, 1, 1))],
        filename="test_reduce_mean.onnx",
    )


def test_reduce_max():
    helper.build_and_convert(
        nodes=[helper.make_node("ReduceMax", ["data"], ["reduced"], axes=[2, 3], keepdims=1)],
        inputs=[helper.make_input("data", (1, 3, 4, 4))],
        outputs=[helper.make_output("reduced", (1, 3, 1, 1))],
        filename="test_reduce_max.onnx",
    )


def test_reduce_min():
    helper.build_and_convert(
        nodes=[helper.make_node("ReduceMin", ["data"], ["reduced"], axes=[2, 3], keepdims=1)],
        inputs=[helper.make_input("data", (1, 3, 4, 4))],
        outputs=[helper.make_output("reduced", (1, 3, 1, 1))],
        filename="test_reduce_min.onnx",
    )


def test_argmax():
    # ArgMax outputs int64 in ONNX; declare correct output dtype
    helper.build_and_convert(
        nodes=[helper.make_node("ArgMax", ["data"], ["reduced"], axis=1, keepdims=1)],
        inputs=[helper.make_input("data", (1, 3, 4, 4))],
        outputs=[helper.make_output("reduced", (1, 1, 4, 4), dtype=helper.TensorProto.INT64)],
        filename="test_argmax.onnx",
    )


def test_argmin():
    helper.build_and_convert(
        nodes=[helper.make_node("ArgMin", ["data"], ["reduced"], axis=1, keepdims=1)],
        inputs=[helper.make_input("data", (1, 3, 4, 4))],
        outputs=[helper.make_output("reduced", (1, 1, 4, 4), dtype=helper.TensorProto.INT64)],
        filename="test_argmin.onnx",
    )


def test_matmul():
    # MatMul: A(K) x B(K, N) -> Y(N)
    w = np.random.randn(4, 8).astype(np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("MatMul", ["A", "B"], ["Y"])],
        inputs=[helper.make_input("A", (1, 4))],
        outputs=[helper.make_output("Y", (1, 8))],
        initializers=[helper.make_weight("B", w)],
        filename="test_matmul.onnx",
    )
