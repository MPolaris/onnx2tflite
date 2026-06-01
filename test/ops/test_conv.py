"""
Unit tests for convolution operators.

ONNX weight format (NCHW):
  - Conv W: (M, C/group, kH, kW)
  - Conv B: (M,) optional

Reference: asserts/Operators.md for ONNX operator specs.
"""
import numpy as np
import helper


def test_conv2d_basic():
    """Standard Conv2d with padding, no bias."""
    in_c, out_c = 3, 6
    w = np.random.randn(out_c, in_c, 3, 3).astype(np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("Conv", ["X", "W"], ["Y"],
                kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])],
        inputs=[helper.make_input("X", (1, in_c, 32, 32))],
        outputs=[helper.make_output("Y", (1, out_c, 32, 32))],
        initializers=[helper.make_weight("W", w)],
        filename="test_conv2d_basic.onnx",
    )


def test_conv2d_with_bias():
    """Conv2d with bias."""
    in_c, out_c = 3, 6
    w = np.random.randn(out_c, in_c, 3, 3).astype(np.float32)
    b = np.random.randn(out_c).astype(np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("Conv", ["X", "W", "B"], ["Y"],
                kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])],
        inputs=[helper.make_input("X", (1, in_c, 32, 32))],
        outputs=[helper.make_output("Y", (1, out_c, 32, 32))],
        initializers=[helper.make_weight("W", w), helper.make_weight("B", b)],
        filename="test_conv2d_bias.onnx",
    )


def test_conv2d_strided():
    """Conv2d with stride=2."""
    in_c, out_c = 3, 6
    w = np.random.randn(out_c, in_c, 3, 3).astype(np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("Conv", ["X", "W"], ["Y"],
                kernel_shape=[3, 3], strides=[2, 2], pads=[0, 0, 0, 0])],
        inputs=[helper.make_input("X", (1, in_c, 32, 32))],
        outputs=[helper.make_output("Y", (1, out_c, 15, 15))],
        initializers=[helper.make_weight("W", w)],
        filename="test_conv2d_strided.onnx",
    )


def test_conv2d_depthwise():
    """Depthwise Conv: group == in_channels == out_channels."""
    in_c = 4
    # W shape: (M, C/group, kH, kW) = (in_c, 1, 3, 3)
    w = np.random.randn(in_c, 1, 3, 3).astype(np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("Conv", ["X", "W"], ["Y"],
                kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0],
                group=in_c)],
        inputs=[helper.make_input("X", (1, in_c, 16, 16))],
        outputs=[helper.make_output("Y", (1, in_c, 14, 14))],
        initializers=[helper.make_weight("W", w)],
        filename="test_conv2d_depthwise.onnx",
    )


def test_conv2d_group():
    """Group Conv: 1 < group < in_channels.
    Using native_groupconv=True to test native grouped conv.
    """
    in_c, out_c, group = 4, 8, 2
    # W: (M, C/group, kH, kW) = (8, 2, 3, 3)
    w = np.random.randn(out_c, in_c // group, 3, 3).astype(np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("Conv", ["X", "W"], ["Y"],
                kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0],
                group=group)],
        inputs=[helper.make_input("X", (1, in_c, 16, 16))],
        outputs=[helper.make_output("Y", (1, out_c, 14, 14))],
        initializers=[helper.make_weight("W", w)],
        filename="test_conv2d_group.onnx",
        # Use split-based group conv (not native) for broader compatibility
        native_groupconv=False,
    )


def test_conv2d_group_native():
    """Group Conv using native Keras grouped convolution (tflite >= 2.9)."""
    in_c, out_c, group = 4, 8, 2
    w = np.random.randn(out_c, in_c // group, 3, 3).astype(np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("Conv", ["X", "W"], ["Y"],
                kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0],
                group=group)],
        inputs=[helper.make_input("X", (1, in_c, 16, 16))],
        outputs=[helper.make_output("Y", (1, out_c, 14, 14))],
        initializers=[helper.make_weight("W", w)],
        filename="test_conv2d_group_native.onnx",
        native_groupconv=True,
    )


def test_conv_transpose():
    """ConvTranspose2d."""
    in_c, out_c = 3, 6
    w = np.random.randn(in_c, out_c, 3, 3).astype(np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])],
        inputs=[helper.make_input("X", (1, in_c, 8, 8))],
        outputs=[helper.make_output("Y", (1, out_c, 10, 10))],
        initializers=[helper.make_weight("W", w)],
        filename="test_conv_transpose.onnx",
    )
