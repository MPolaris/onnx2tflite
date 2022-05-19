import os
import numpy as np
import tensorflow as tf
from onnx_runner import ONNXModel

def main(onnx_model_path, tflite_model_path, interest_layers = []):
    model_onnx = ONNXModel(onnx_model_path, interest_layers)

    X = np.random.randn(*model_onnx.input_shape).astype(np.float32)*10
    onnx_out = model_onnx.forward(X)[-1]
    if len(interest_layers) == 0:
        print("onnx_out.shape = ", onnx_out.shape)
    else:
        print("inner_layer.shape = ", onnx_out.shape)

    if len(X.shape) > 2:
        _transpose_index = [i for i in range(len(X.shape))]
        _transpose_index = _transpose_index[0:1] + _transpose_index[2:] + _transpose_index[1:2]
        X = X.transpose(*_transpose_index)
    model_tflite = tf.lite.Interpreter(model_path=tflite_model_path)
    model_tflite.allocate_tensors()
    input_details, output_details  = model_tflite.get_input_details(), model_tflite.get_output_details()
    input_shape = input_details[0]['shape']
    model_tflite.set_tensor(input_details[0]['index'], X)
    model_tflite.invoke()
    tflite_output = model_tflite.get_tensor(output_details[0]['index'])
    print("tflite_out.shape = ", tflite_output.shape)
    if len(tflite_output.shape) > 2 or onnx_out.shape != tflite_output.shape:
        shape = [i for i in range(len(tflite_output.shape))]
        newshape = [shape[0], shape[-1], *shape[1:-1]]
        tflite_output = tflite_output.transpose(*newshape)
    print("tflite_out_reshape.shape = ", tflite_output.shape)
    assert len(onnx_out) == len(tflite_output) and onnx_out.shape == tflite_output.shape, "输出不一致"
    diff = np.abs(onnx_out - tflite_output)
    mean = np.mean(diff)
    std = np.std(diff, ddof=1)
    max = np.max(mean)
    print("差值平均值:{:^9.9f}, 方差:{:^9.9f}, 差值最大值:{:^9.9f}".format(mean, std, max))
    return [mean, max]

if __name__ == "__main__":
    main(onnx_model_path = "./models/shufflenet.onnx",
            tflite_model_path = "./models/shufflenet.tflite",
            interest_layers = []
            )