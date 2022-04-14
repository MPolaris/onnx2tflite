import os
import logging
import argparse
from utils import load_onnx_modelproto, keras_builder, tflite_builder

LOG = logging.getLogger("converter running:")

def onnx_converter(onnx_model_path:str, need_simplify:bool=True, tflite_out_path:str=None,
                weight_quant:bool=False, int8_model:bool=False, image_root:str=None):

    model_proto = load_onnx_modelproto(onnx_model_path, need_simplify)

    keras_model = keras_builder(model_proto)

    tflite_model = tflite_builder(keras_model, weight_quant, int8_model, image_root)

    if tflite_out_path is None:
        tflite_out_path, tflite_model_name = os.path.split(onnx_model_path)
        tflite_out_path = os.path.join(tflite_out_path, "".join(tflite_model_name.split('.')[:-1]) + ".tflite")
    if os.path.isdir(tflite_out_path):
        tflite_out_path = os.path.join(tflite_out_path, os.path.split(onnx_model_path)[-1].split(".")[0] + ".tflite")
    with open(tflite_out_path, "wb") as fp:
        fp.write(tflite_model)
    
    LOG.info(f"转换完成, 模型保存在{tflite_out_path}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./models/mobilenetv3.onnx', help='onnx model path')
    parser.add_argument('--outpath', type=str, default='./utils', help='tflite model save path')
    parser.add_argument('--simplify', default=True, action='store_true', help='onnx model need simplify model')
    parser.add_argument('--weigthquant', default=False, action='store_true', help='tflite weigth int8 quant')
    parser.add_argument('--int8', default=False, action='store_true', help='tflite weigth int8 quant, include input output')
    parser.add_argument('--imgroot', type=str, default=None, help='when int8=True, imgroot should give for calculating mean and norm')
    opt = parser.parse_args()
    return opt

def run():
    opt = parse_opt()
    onnx_converter(
        onnx_model_path = opt.weights,
        need_simplify = opt.simplify,
        tflite_out_path = opt.outpath,
        weight_quant=opt.weigthquant,
        int8_model=opt.int8,
        image_root=opt.imgroot
    )

if __name__ == "__main__":
    run()