"""
Shared utilities for building ONNX IR graphs and running conversion tests.

Reference: asserts/Operators.md for ONNX operator specs (inputs, outputs, attributes).
"""
import os
import numpy as np
import onnx
from onnx import helper as onnx_helper, numpy_helper, TensorProto  # noqa: F401 - TensorProto exported for test use

from onnx2tflite import onnx_converter

MODEL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unit_test")
os.makedirs(MODEL_ROOT, exist_ok=True)

# Ops that cause segfault in TF 2.12 TFLite interpreter — skip direct-IR
_DIRECT_IR_CRASH_OPS = set()



def make_input(name, shape, dtype=TensorProto.FLOAT):
    """Create a ValueInfoProto for a graph input."""
    return onnx_helper.make_tensor_value_info(name, dtype, shape)


def make_output(name, shape, dtype=TensorProto.FLOAT):
    """Create a ValueInfoProto for a graph output."""
    return onnx_helper.make_tensor_value_info(name, dtype, shape)


def make_weight(name, np_array):
    """Create a TensorProto initializer from a numpy array."""
    return numpy_helper.from_array(np_array, name=name)


def make_node(op_type, inputs, outputs, **attributes):
    """Create a NodeProto with the given op type and attributes."""
    return onnx_helper.make_node(op_type, inputs, outputs, **attributes)


def build_and_convert(nodes, inputs, outputs, initializers=None, filename="test.onnx",
                      opset_version=14, use_direct_ir=None, **overrides):
    """Build an ONNX model, convert to TFLite, and assert error < 1e-3.

    Tests both Keras and direct-IR paths by default (use_direct_ir=None).
    Set use_direct_ir=False to test only Keras path, or True for only direct-IR.
    """
    graph = onnx_helper.make_graph(
        nodes=nodes, name="test_graph", inputs=inputs, outputs=outputs,
        initializer=initializers or [],
    )
    opset_imports = [onnx_helper.make_opsetid("", opset_version)]
    model = onnx_helper.make_model(graph, opset_imports=opset_imports)
    path = os.path.join(MODEL_ROOT, filename)
    onnx.save(model, path)

    base_kwargs = dict(
        need_simplify=False, output_path=MODEL_ROOT, target_formats=["tflite"],
        native_groupconv=False, fp16_model=False, int8_model=False,
    )
    base_kwargs.update(overrides)

    paths_to_test = []
    if use_direct_ir is None or use_direct_ir is False:
        paths_to_test.append(("Keras", False))
    if use_direct_ir is None or use_direct_ir is True:
        paths_to_test.append(("direct-IR", True))

    for label, direct in paths_to_test:
        kwargs = dict(base_kwargs, use_direct_ir=direct)
        if direct:
            kwargs.pop("native_groupconv", None)
        # Skip ops that cause native crash in TF 2.12 interpreter
        if direct and any(n.op_type in _DIRECT_IR_CRASH_OPS for n in nodes):
            continue
        try:
            result = onnx_converter(onnx_model_path=path, **kwargs)
        except NotImplementedError as e:
            import warnings
            warnings.warn(f"Skipping {label} for {filename}: {e}")
            continue
        error = result.get("tflite_error", None)
        assert error is not None, f"No tflite_error ({label}) for {filename}"
        assert error < 1e-3, f"tflite_error={error} >= 1e-3 for {filename} ({label} path)"

    return result
