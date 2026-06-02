"""TFLite IR direct builder — bypasses Keras for ONNX→TFLite conversion."""
from onnx2tflite.components.tflite_ir.builder import TFLiteBuilder, Layout
from onnx2tflite.components.tflite_ir.op_mapping import dispatch

# Force-load all handler modules to register operators
from onnx2tflite.components.tflite_ir import math
from onnx2tflite.components.tflite_ir import convolution
from onnx2tflite.components.tflite_ir import pooling
from onnx2tflite.components.tflite_ir import deformation
from onnx2tflite.components.tflite_ir import normalization
from onnx2tflite.components.tflite_ir import reduction
from onnx2tflite.components.tflite_ir import decompose


def build_tflite_ir(model_proto) -> bytes:
    """Build a TFLite FlatBuffer from an ONNX ModelProto.

    Walks the ONNX graph node-by-node, dispatching each operator to a handler
    that emits the corresponding TFLite IR directly.

    Args:
        model_proto: onnx.ModelProto (loaded and optionally simplified)

    Returns:
        bytes: The serialized .tflite model
    """
    builder = TFLiteBuilder(model_proto)

    for node in model_proto.graph.node:
        dispatch(builder, node)

    # Mark graph outputs (after all nodes processed)
    for out_info in model_proto.graph.output:
        if out_info.name in builder._tensor_map:
            builder.mark_output(builder._tensor_map[out_info.name])

    return builder.build()
