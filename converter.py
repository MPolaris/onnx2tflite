import os
import logging
import argparse
from utils import load_onnx_modelproto, keras_builder, tflite_builder

LOG = logging.getLogger("converter running:")

def onnx_converter(onnx_model_path:str, need_simplify:bool=True, output_path:str=None, target_formats:list = ['keras', 'tflite'],
                weight_quant:bool=False, int8_model:bool=False, image_root:str=None):
    if not isinstance(target_formats, list) and  'keras' not in target_formats and 'tflite' not in target_formats:
        raise KeyError("'keras' or 'tflite' should in list")
    
    model_proto = load_onnx_modelproto(onnx_model_path, need_simplify)

    keras_model = keras_builder(model_proto)

    if 'tflite' in target_formats:
        tflite_model = tflite_builder(keras_model, weight_quant, int8_model, image_root)

    onnx_path, model_name = os.path.split(onnx_model_path)
    if output_path is None:
        output_path = onnx_path
    output_path = os.path.join(output_path, model_name.split('.')[0])

    if 'keras' in target_formats:
        keras_model.save(output_path + ".h5")
        LOG.info(f"keras model saved in {output_path}.h5")

    if 'tflite' in target_formats:
        with open(output_path + ".tflite", "wb") as fp:
            fp.write(tflite_model)
        LOG.info(f"tflite model saved in {output_path}.tflite")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./models/mobilenetv3.onnx', help='onnx model path')
    parser.add_argument('--outpath', type=str, default=None, help='tflite model save path')
    parser.add_argument('--simplify', default=True, action='store_true', help='onnx model need simplify model')
    parser.add_argument('--weigthquant', default=False, action='store_true', help='tflite weigth int8 quant')
    parser.add_argument('--int8', default=False, action='store_true', help='tflite weigth int8 quant, include input output')
    parser.add_argument('--imgroot', type=str, default=None, help='when int8=True, imgroot should give for calculating mean and norm')
    parser.add_argument('--formats', nargs='+', default=['keras', 'tflite'], help='available formats are (h5, tflite)')
    opt = parser.parse_args()
    return opt

def run():
    opt = parse_opt()
    onnx_converter(
        onnx_model_path = opt.weights,
        need_simplify = opt.simplify,
        output_path = opt.outpath,
        target_formats = opt.formats,
        weight_quant=opt.weigthquant,
        int8_model=opt.int8,
        image_root=opt.imgroot
    )

if __name__ == "__main__":
    run()