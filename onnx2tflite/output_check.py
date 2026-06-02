import os
import numpy as np
import tensorflow as tf
import onnxruntime as ort
from onnx2tflite.utils.definitions import Layout
from onnx2tflite.utils.dimension_utils import tensor_NDC_to_NCD_format


def _run_onnx(onnx_proto):
    """Run ONNX model with ones-input and return outputs."""
    sess = ort.InferenceSession(onnx_proto.SerializeToString())
    inputs = {}
    for inp in sess.get_inputs():
        shape = [d if (isinstance(d, int) and d > 0) else 1 for d in inp.shape]
        inputs[inp.name] = np.ones(shape, dtype=np.float32)
    return sess.run([], inputs)


def _run_tflite_file(model_path: str) -> np.ndarray:
    """Run TFLite model from file path with ones-input, return first output."""
    interp = tf.lite.Interpreter(model_path=model_path, num_threads=4)
    interp.allocate_tensors()
    for d in interp.get_input_details():
        interp.set_tensor(d['index'], np.ones(d['shape'], dtype=np.float32))
    interp.invoke()
    return interp.get_tensor(interp.get_output_details()[0]['index'])


def _run_tflite_bytes(tflite_bytes: bytes):
    """Run TFLite model from bytes with ones-input, return all outputs as list."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    for d in interp.get_input_details():
        interp.set_tensor(d['index'], np.ones(d['shape'], dtype=np.float32))
    interp.invoke()
    return [interp.get_tensor(d['index']) for d in interp.get_output_details()]


def _compute_max_error(onnx_outputs, target_outputs):
    """Compare ONNX outputs against target outputs, auto-handle NHWC→NCHW.
    Returns MIN error across output pairs (best shape-matched pair)."""
    best = float('inf')
    for onnx_out in onnx_outputs:
        for tgt_out in target_outputs:
            if onnx_out.shape != tgt_out.shape and tgt_out.ndim >= 3:
                perm = [0, tgt_out.ndim - 1] + list(range(1, tgt_out.ndim - 1))
                tgt_out = np.transpose(tgt_out, perm)
            if onnx_out.shape == tgt_out.shape:
                best = min(best, np.max(np.abs(onnx_out - tgt_out)))
    return 0.0 if best == float('inf') else best


def check_tflite_error(onnx_proto, tflite_bytes: bytes) -> float:
    """Compare ONNX model output vs TFLite model (given as bytes).
    Returns max element-wise error. Auto-handles NHWC↔NCHW layout."""
    onnx_outputs = _run_onnx(onnx_proto)
    tflite_outputs = _run_tflite_bytes(tflite_bytes)
    return _compute_max_error(onnx_outputs, tflite_outputs)


def get_elements_error(onnx_proto, keras_model_path: str, tflite_model_path: str,
                       input_layout: dict, output_layout: dict) -> dict:
    """Compare ONNX model against Keras + TFLite models (from file paths).
    Returns {'keras': max_error, 'tflite': max_error}."""
    onnx_outputs = _run_onnx(onnx_proto)
    result = {}

    channel_last = any(output_layout.get(o.name) == Layout.Channel_Last
                       for o in onnx_proto.graph.output)

    if keras_model_path is not None:
        keras_runtime = tf.keras.models.load_model(keras_model_path)
        keras_input = [np.ones(list(s.shape), dtype=np.float32) for s in keras_runtime.inputs]
        keras_out = keras_runtime.predict(keras_input, verbose=0)
        if not isinstance(keras_out, list):
            keras_out = [keras_out]
        if channel_last:
            keras_out = [tensor_NDC_to_NCD_format(o) if o.ndim >= 3 else o for o in keras_out]
        result['keras'] = _compute_max_error(onnx_outputs, keras_out)

    if tflite_model_path is not None:
        tflite_out = _run_tflite_file(tflite_model_path)
        if channel_last:
            tflite_out = tensor_NDC_to_NCD_format(tflite_out)
        result['tflite'] = _compute_max_error(onnx_outputs, [tflite_out])

    return result