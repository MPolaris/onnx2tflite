"""
Unit tests for activation operators.

Reference: asserts/Operators.md for ONNX operator specs.
"""
import numpy as np
import helper


def test_relu():
    helper.build_and_convert(
        nodes=[helper.make_node("Relu", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_relu.onnx",
    )


def test_sigmoid():
    helper.build_and_convert(
        nodes=[helper.make_node("Sigmoid", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_sigmoid.onnx",
    )


def test_tanh():
    helper.build_and_convert(
        nodes=[helper.make_node("Tanh", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_tanh.onnx",
    )


def test_hard_swish():
    helper.build_and_convert(
        nodes=[helper.make_node("HardSwish", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_hardswish.onnx",
    )


def test_sin():
    helper.build_and_convert(
        nodes=[helper.make_node("Sin", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_sin.onnx",
    )


def test_cos():
    helper.build_and_convert(
        nodes=[helper.make_node("Cos", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_cos.onnx",
    )


def test_tan():
    helper.build_and_convert(
        nodes=[helper.make_node("Tan", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_tan.onnx",
    )


def test_sinh():
    helper.build_and_convert(
        nodes=[helper.make_node("Sinh", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_sinh.onnx",
    )


def test_cosh():
    helper.build_and_convert(
        nodes=[helper.make_node("Cosh", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_cosh.onnx",
    )


def test_softplus():
    helper.build_and_convert(
        nodes=[helper.make_node("Softplus", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_softplus.onnx",
    )


def test_softsign():
    helper.build_and_convert(
        nodes=[helper.make_node("Softsign", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_softsign.onnx",
    )


def test_selu():
    helper.build_and_convert(
        nodes=[helper.make_node("Selu", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_selu.onnx",
    )


def test_elu():
    helper.build_and_convert(
        nodes=[helper.make_node("Elu", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_elu.onnx",
    )


def test_hard_sigmoid():
    # alpha=0.2, beta=0.5 are defaults per ONNX spec
    helper.build_and_convert(
        nodes=[helper.make_node("HardSigmoid", ["X"], ["Y"], alpha=0.2, beta=0.5)],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_hardsigmoid.onnx",
    )


def test_leaky_relu():
    # alpha defaults to 0.01 per ONNX spec
    helper.build_and_convert(
        nodes=[helper.make_node("LeakyRelu", ["X"], ["Y"], alpha=0.01)],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_leakyrelu.onnx",
    )


def test_celu():
    # alpha defaults to 1.0 per ONNX spec
    helper.build_and_convert(
        nodes=[helper.make_node("Celu", ["X"], ["Y"], alpha=1.0)],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        filename="test_celu.onnx",
    )


def test_softmax():
    # ONNX Softmax uses "input" (lowercase) as input name, axis defaults to -1
    helper.build_and_convert(
        nodes=[helper.make_node("Softmax", ["input"], ["output"], axis=-1)],
        inputs=[helper.make_input("input", (1, 3, 4, 4))],
        outputs=[helper.make_output("output", (1, 3, 4, 4))],
        filename="test_softmax.onnx",
    )


def test_prelu():
    # PRelu slope: for NCHW input (1,3,H,W), slope must broadcast to (1,3,1,1)
    slope = np.array([0.25, 0.5, 0.75], dtype=np.float32).reshape(3, 1, 1)
    helper.build_and_convert(
        nodes=[helper.make_node("PRelu", ["X", "slope"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        initializers=[helper.make_weight("slope", slope)],
        filename="test_prelu.onnx",
    )
