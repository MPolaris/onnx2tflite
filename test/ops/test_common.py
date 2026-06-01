"""
Unit tests for common/normalization/pooling operators.

Reference: asserts/Operators.md for ONNX operator specs.
"""
import numpy as np
import helper


# --- Normalization ---

def test_batch_normalization():
    # ONNX inputs: X, scale, B, input_mean, input_var
    channels = 3
    scale = np.ones(channels, dtype=np.float32)
    bias = np.zeros(channels, dtype=np.float32)
    mean = np.zeros(channels, dtype=np.float32)
    var = np.ones(channels, dtype=np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("BatchNormalization", ["X", "scale", "B", "mean", "var"], ["Y"], epsilon=1e-5, momentum=0.9)],
        inputs=[helper.make_input("X", (1, channels, 4, 4))],
        outputs=[helper.make_output("Y", (1, channels, 4, 4))],
        initializers=[helper.make_weight("scale", scale), helper.make_weight("B", bias),
                      helper.make_weight("mean", mean), helper.make_weight("var", var)],
        filename="test_batchnorm.onnx",
    )


def test_instance_normalization():
    channels = 3
    scale = np.ones(channels, dtype=np.float32)
    bias = np.zeros(channels, dtype=np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("InstanceNormalization", ["X", "scale", "B"], ["Y"], epsilon=1e-5)],
        inputs=[helper.make_input("X", (1, channels, 4, 4))],
        outputs=[helper.make_output("Y", (1, channels, 4, 4))],
        initializers=[helper.make_weight("scale", scale), helper.make_weight("B", bias)],
        filename="test_instancenorm.onnx",
    )


# --- Pad ---

def test_pad():
    # pads format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    # For NCHW 4D: pads = [0, 0, 1, 1, 0, 0, 1, 1] (pad H and W by 1)
    # Opset 11+: pads is an INPUT tensor, not attribute
    pads_val = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    helper.build_and_convert(
        nodes=[helper.make_node("Pad", ["X", "pads"], ["Y"], mode="constant")],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 6, 6))],
        initializers=[helper.make_weight("pads", pads_val)],
        filename="test_pad.onnx",
    )


# --- Clip ---

def test_clip():
    # Opset 11+: min/max are optional INPUTS, not attributes
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(6.0, dtype=np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("Clip", ["input", "min", "max"], ["output"])],
        inputs=[helper.make_input("input", (1, 3, 4, 4))],
        outputs=[helper.make_output("output", (1, 3, 4, 4))],
        initializers=[helper.make_weight("min", min_val), helper.make_weight("max", max_val)],
        filename="test_clip.onnx",
    )


# --- Pooling ---

def test_average_pool():
    helper.build_and_convert(
        nodes=[helper.make_node("AveragePool", ["X"], ["Y"],
                kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 2, 2))],
        filename="test_avgpool.onnx",
    )


def test_max_pool():
    helper.build_and_convert(
        nodes=[helper.make_node("MaxPool", ["X"], ["Y"],
                kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 2, 2))],
        filename="test_maxpool.onnx",
    )


def test_global_max_pool():
    helper.build_and_convert(
        nodes=[helper.make_node("GlobalMaxPool", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 1, 1))],
        filename="test_global_maxpool.onnx",
    )


def test_global_average_pool():
    helper.build_and_convert(
        nodes=[helper.make_node("GlobalAveragePool", ["X"], ["Y"])],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 1, 1))],
        filename="test_global_avgpool.onnx",
    )


# --- Upsample / Resize ---

def test_upsample():
    # Upsample is deprecated in opset 10+; use opset 9
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("Upsample", ["X", "scales"], ["Y"], mode="nearest")],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 8, 8))],
        initializers=[helper.make_weight("scales", scales)],
        filename="test_upsample.onnx",
        opset_version=9,
    )


def test_resize():
    # Resize with roi (empty) and scales inputs (opset 13+ format)
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    roi = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("Resize", ["X", "roi", "scales"], ["Y"], mode="nearest")],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 8, 8))],
        initializers=[
            helper.make_weight("roi", roi),
            helper.make_weight("scales", scales),
        ],
        filename="test_resize.onnx",
    )


# --- Gemm ---

def test_gemm():
    # Gemm: Y = alpha * A * B + beta * C
    # A: (M, K), B: (K, N), C: broadcastable to (M, N)
    w = np.random.randn(4, 8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("Gemm", ["A", "B", "C"], ["Y"], alpha=1.0, beta=1.0, transA=0, transB=0)],
        inputs=[helper.make_input("A", (1, 4))],
        outputs=[helper.make_output("Y", (1, 8))],
        initializers=[helper.make_weight("B", w), helper.make_weight("C", b)],
        filename="test_gemm.onnx",
    )


# --- ScatterND ---

def test_scatter_nd():
    data = np.random.randn(1, 3, 4, 4).astype(np.float32)
    indices = np.array([[0, 1, 2, 2]], dtype=np.int64)
    updates = np.random.randn(1).astype(np.float32)
    helper.build_and_convert(
        nodes=[helper.make_node("ScatterND", ["data", "indices", "updates"], ["Y"])],
        inputs=[helper.make_input("data", (1, 3, 4, 4))],
        outputs=[helper.make_output("Y", (1, 3, 4, 4))],
        initializers=[helper.make_weight("indices", indices), helper.make_weight("updates", updates)],
        filename="test_scatternd.onnx",
    )


# --- Passthrough ops ---

def test_identity():
    helper.build_and_convert(
        nodes=[helper.make_node("Identity", ["input"], ["output"])],
        inputs=[helper.make_input("input", (1, 3, 4, 4))],
        outputs=[helper.make_output("output", (1, 3, 4, 4))],
        filename="test_identity.onnx",
    )


def test_dropout():
    # Dropout is ignored during inference
    helper.build_and_convert(
        nodes=[helper.make_node("Dropout", ["data"], ["output"])],
        inputs=[helper.make_input("data", (1, 3, 4, 4))],
        outputs=[helper.make_output("output", (1, 3, 4, 4))],
        filename="test_dropout.onnx",
    )


# --- TopK ---

def test_topk():
    # ONNX TopK: Values has input dtype, Indices is int64
    K = np.array([3], dtype=np.int64)
    helper.build_and_convert(
        nodes=[helper.make_node("TopK", ["X", "K"], ["Values", "Indices"], axis=-1, largest=1, sorted=1)],
        inputs=[helper.make_input("X", (1, 3, 4, 4))],
        outputs=[helper.make_output("Values", (1, 3, 4, 3), dtype=helper.TensorProto.FLOAT),
                 helper.make_output("Indices", (1, 3, 4, 3), dtype=helper.TensorProto.INT64)],
        initializers=[helper.make_weight("K", K)],
        filename="test_topk.onnx",
    )


# --- Cast ---

def test_cast():
    # to=6 means int32; declare correct output dtype
    helper.build_and_convert(
        nodes=[helper.make_node("Cast", ["input"], ["output"], to=6)],
        inputs=[helper.make_input("input", (1, 3, 4, 4))],
        outputs=[helper.make_output("output", (1, 3, 4, 4), dtype=helper.TensorProto.INT32)],
        filename="test_cast.onnx",
    )
