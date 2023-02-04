import os
import numpy as np
import tensorflow as tf
import onnxruntime as ort

def get_elements_error(onnx_proto, tflite_model_path:str):
    # test onnx
    onnx_runtime = ort.InferenceSession(onnx_proto.SerializeToString())
    onnx_inputs, tflite_inputs = {}, []
    for inp in onnx_runtime.get_inputs():
        shape = inp.shape
        if isinstance(shape[0], str) or shape[0] < 1:
            shape[0] = 1
        onnx_inputs[inp.name] = np.ones(shape, dtype=np.float32)
        if len(shape) > 2:
            _transpose_index = [i for i in range(len(shape))]
            _transpose_index = _transpose_index[0:1] + _transpose_index[2:] + _transpose_index[1:2]
            tflite_inputs.append(onnx_inputs[inp.name].copy().transpose(*_transpose_index))
        else:
            tflite_inputs.append(onnx_inputs[inp.name].copy())
    onnx_outputs = onnx_runtime.run([], onnx_inputs)

    # test tflite
    tflite_runtime = tf.lite.Interpreter(tflite_model_path, num_threads=4)
    tflite_runtime.allocate_tensors()
    input_details, output_details  = tflite_runtime.get_input_details(), tflite_runtime.get_output_details()
    for i in range(len(tflite_inputs)):
        tflite_runtime.set_tensor(input_details[i]['index'], tflite_inputs[i])
    tflite_runtime.invoke()

    # only compare one output is ok.
    tflite_output = tflite_runtime.get_tensor(output_details[0]['index'])
    if len(tflite_output.shape) > 2:
        shape = [i for i in range(len(tflite_output.shape))]
        newshape = [shape[0], shape[-1], *shape[1:-1]]
        tflite_output = tflite_output.transpose(*newshape)

    # get max error
    max_error = 1000
    for onnx_output in onnx_outputs:
        if onnx_output.shape != tflite_output.shape:
            continue
        diff = np.abs(onnx_output - tflite_output)
        max_diff = np.max(diff)
        max_error = min(max_error, max_diff)
    
    return max_error
# onnx_model_path, tflite_model_path = "./models/yolov7_sub.onnx", "./models/yolov7.tflite"