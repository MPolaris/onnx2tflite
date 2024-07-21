import os
import numpy as np
import tensorflow as tf
import onnxruntime as ort
from onnx2tflite.utils.definitions import Layout
from onnx2tflite.utils.dimension_utils import tensor_NDC_to_NCD_format

def tflite_run(model_path:str) -> np.ndarray:
    '''
        tflite runtime
    '''
    tflite_runtime = tf.lite.Interpreter(model_path, num_threads=4)
    tflite_runtime.allocate_tensors()
    input_details, output_details  = tflite_runtime.get_input_details(), tflite_runtime.get_output_details()
    for i in range(len(input_details)):
        tflite_runtime.set_tensor(input_details[i]['index'], np.ones(input_details[i]['shape'], dtype=np.float32))
    tflite_runtime.invoke()

    # only compare one output is ok.
    tflite_output = tflite_runtime.get_tensor(output_details[0]['index'])
    return tflite_output

def keras_run(model_path:str) -> np.ndarray:
    '''
        keras runtime
    '''
    keras_runtime = tf.keras.models.load_model(model_path)
    _input = []
    for inp in keras_runtime.inputs:
        _input.append(np.ones(list(inp.shape), dtype=np.float32))

    keras_output = keras_runtime.predict(_input)
    # only compare one output is ok.
    if isinstance(keras_output, list):
        keras_output = keras_output[0]
    return keras_output
    
        
def get_elements_error(onnx_proto, keras_model_path:str, tflite_model_path:str, input_layout:dict, output_layout:dict) -> dict:
    '''
        use ones input arr to check model.
        more carefully check is up to youself custom code.
    '''
    result = {}
    # test onnx
    onnx_runtime = ort.InferenceSession(onnx_proto.SerializeToString())
    onnx_inputs = {}
    for inp in onnx_runtime.get_inputs():
        shape = inp.shape
        if isinstance(shape[0], str) or shape[0] < 1:
            shape[0] = 1
        onnx_inputs[inp.name] = np.ones(shape, dtype=np.float32)
        if len(shape) > 2:
            _transpose_index = [i for i in range(len(shape))]
            _transpose_index = _transpose_index[0:1] + _transpose_index[2:] + _transpose_index[1:2]
    onnx_outputs = onnx_runtime.run([], onnx_inputs)

    channel_last = False
    for oup in onnx_proto.graph.output:
        channel_last = output_layout[oup.name] == Layout.Channel_Last
        break

    if keras_model_path is not None:
        # test keras model
        keras_output = keras_run(keras_model_path)
        if channel_last:
            keras_output = tensor_NDC_to_NCD_format(keras_output)
        # get max error
        keras_max_error = 1000
        for onnx_output in onnx_outputs:
            if onnx_output.shape != keras_output.shape:
                continue
            diff = np.abs(onnx_output - keras_output)
            max_diff = np.max(diff)
            keras_max_error = min(keras_max_error, max_diff)
        result['keras'] = keras_max_error

    if tflite_model_path is not None:
        # test tflite
        tflite_output = tflite_run(tflite_model_path)
        if channel_last:
            tflite_output = tensor_NDC_to_NCD_format(tflite_output)
        # get max error
        tflite_max_error = 1000
        for onnx_output in onnx_outputs:
            if onnx_output.shape != tflite_output.shape:
                continue
            diff = np.abs(onnx_output - tflite_output)
            max_diff = np.max(diff)
            tflite_max_error = min(tflite_max_error, max_diff)
        result['tflite'] = tflite_max_error
    
    return result