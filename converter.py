import os
import numpy as np
from utils import LOG, load_onnx_modelproto, keras_builder, tflite_builder

def onnx_converter(onnx_model_path:str, need_simplify:bool=True, tflite_out_path:str=None,
                weight_quant:bool=False, int8_model:bool=False, image_root:str=None):

    model_proto = load_onnx_modelproto(onnx_model_path, need_simplify)

    keras_model = keras_builder(model_proto)

    tflite_model = tflite_builder(keras_model, weight_quant, int8_model, image_root)

    if tflite_out_path is None:
        tflite_model_name = os.path.split(onnx_model_path)[-1]
        tflite_out_path = "".join(tflite_model_name.split('.')[:-1]) + ".tflite"
    with open(tflite_out_path, "wb") as fp:
        fp.write(tflite_model)
    
    LOG.info(f"转换完成, 模型保存在{tflite_out_path}")


if __name__ == "__main__":
    onnx_converter(
        onnx_model_path = "./mobilenetv3.onnx",
        need_simplify = True,
        tflite_out_path = "./mobilenetv3.tflite"
    )