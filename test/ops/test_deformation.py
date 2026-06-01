"""
Unit tests for deformation/shape-manipulation operators.

Reference: asserts/Operators.md for ONNX operator specs.
"""
import numpy as np
import helper


# --- Transpose ---

def test_transpose():
    # perm: NCHW -> NHWC = [0, 2, 3, 1]
    helper.build_and_convert(
        nodes=[helper.make_node("Transpose", ["data"], ["transposed"], perm=[0, 2, 3, 1])],
        inputs=[helper.make_input("data", (1, 3, 4, 4))],
        outputs=[helper.make_output("transposed", (1, 4, 4, 3))],
        filename="test_transpose.onnx",
    )


# --- Slice ---

def test_slice():
    # Opset 10+: Slice uses inputs for starts, ends, axes, steps
    starts = np.array([0], dtype=np.int64)
    ends = np.array([2], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    helper.build_and_convert(
        nodes=[helper.make_node("Slice", ["data", "starts", "ends", "axes"], ["sliced"])],
        inputs=[helper.make_input("data", (1, 3, 4, 4))],
        outputs=[helper.make_output("sliced", (1, 2, 4, 4))],
        initializers=[helper.make_weight("starts", starts), helper.make_weight("ends", ends),
                      helper.make_weight("axes", axes)],
        filename="test_slice.onnx",
    )


# --- Gather ---

def test_gather():
    indices = np.array([0, 2], dtype=np.int64)
    helper.build_and_convert(
        nodes=[helper.make_node("Gather", ["data", "indices"], ["output"], axis=1)],
        inputs=[helper.make_input("data", (1, 3, 4, 4))],
        outputs=[helper.make_output("output", (1, 2, 4, 4))],
        initializers=[helper.make_weight("indices", indices)],
        filename="test_gather.onnx",
    )


# --- Concat ---

def test_concat():
    helper.build_and_convert(
        nodes=[helper.make_node("Concat", ["A", "B", "C"], ["output"], axis=1)],
        inputs=[
            helper.make_input("A", (1, 2, 4, 4)),
            helper.make_input("B", (1, 3, 4, 4)),
            helper.make_input("C", (1, 1, 4, 4)),
        ],
        outputs=[helper.make_output("output", (1, 6, 4, 4))],
        filename="test_concat.onnx",
    )


# --- Reshape ---

def test_reshape():
    shape = np.array([1, 48], dtype=np.int64)
    helper.build_and_convert(
        nodes=[helper.make_node("Reshape", ["data", "shape"], ["reshaped"])],
        inputs=[helper.make_input("data", (1, 3, 4, 4))],
        outputs=[helper.make_output("reshaped", (1, 48))],
        initializers=[helper.make_weight("shape", shape)],
        filename="test_reshape.onnx",
    )


# --- Flatten ---

def test_flatten():
    # axis=1 means flatten starting from dim 1
    helper.build_and_convert(
        nodes=[helper.make_node("Flatten", ["input"], ["output"], axis=1)],
        inputs=[helper.make_input("input", (1, 3, 4, 4))],
        outputs=[helper.make_output("output", (1, 48))],
        filename="test_flatten.onnx",
    )


# --- Split ---

def test_split():
    # split attribute specifies the size of each output along axis
    split_val = np.array([2, 1], dtype=np.int64)
    helper.build_and_convert(
        nodes=[helper.make_node("Split", ["input", "split"], ["out1", "out2"], axis=1)],
        inputs=[helper.make_input("input", (1, 3, 4, 4))],
        outputs=[helper.make_output("out1", (1, 2, 4, 4)), helper.make_output("out2", (1, 1, 4, 4))],
        initializers=[helper.make_weight("split", split_val)],
        filename="test_split.onnx",
    )


# --- Expand ---

def test_expand():
    shape = np.array([1, 3, 8, 4], dtype=np.int64)
    helper.build_and_convert(
        nodes=[helper.make_node("Expand", ["input", "shape"], ["output"])],
        inputs=[helper.make_input("input", (1, 3, 1, 4))],
        outputs=[helper.make_output("output", (1, 3, 8, 4))],
        initializers=[helper.make_weight("shape", shape)],
        filename="test_expand.onnx",
    )


# --- GatherElements ---

def test_gather_elements():
    indices = np.zeros((1, 1, 4, 4), dtype=np.int64)
    helper.build_and_convert(
        nodes=[helper.make_node("GatherElements", ["data", "indices"], ["output"], axis=1)],
        inputs=[helper.make_input("data", (1, 3, 4, 4))],
        outputs=[helper.make_output("output", (1, 1, 4, 4))],
        initializers=[helper.make_weight("indices", indices)],
        filename="test_gather_elements.onnx",
    )


# --- Tile ---

def test_tile():
    repeats = np.array([1, 2, 1, 1], dtype=np.int64)
    helper.build_and_convert(
        nodes=[helper.make_node("Tile", ["input", "repeats"], ["output"])],
        inputs=[helper.make_input("input", (1, 3, 4, 4))],
        outputs=[helper.make_output("output", (1, 6, 4, 4))],
        initializers=[helper.make_weight("repeats", repeats)],
        filename="test_tile.onnx",
    )


# --- Unsqueeze ---

def test_unsqueeze():
    # axes as initializer (second input)
    axes = np.array([1], dtype=np.int64)
    helper.build_and_convert(
        nodes=[helper.make_node("Unsqueeze", ["data", "axes"], ["expanded"])],
        inputs=[helper.make_input("data", (1, 3, 4, 4))],
        outputs=[helper.make_output("expanded", (1, 1, 3, 4, 4))],
        initializers=[helper.make_weight("axes", axes)],
        filename="test_unsqueeze.onnx",
    )


# --- Squeeze ---

def test_squeeze():
    axes = np.array([1], dtype=np.int64)
    helper.build_and_convert(
        nodes=[helper.make_node("Squeeze", ["data", "axes"], ["squeezed"])],
        inputs=[helper.make_input("data", (1, 1, 4, 4))],
        outputs=[helper.make_output("squeezed", (1, 4, 4))],
        initializers=[helper.make_weight("axes", axes)],
        filename="test_squeeze.onnx",
    )


# --- DepthToSpace ---

def test_depth_to_space():
    # blocksize=2, mode="DCR" (default)
    helper.build_and_convert(
        nodes=[helper.make_node("DepthToSpace", ["input"], ["output"], blocksize=2, mode="DCR")],
        inputs=[helper.make_input("input", (1, 12, 4, 4))],
        outputs=[helper.make_output("output", (1, 3, 8, 8))],
        filename="test_depth_to_space.onnx",
    )
