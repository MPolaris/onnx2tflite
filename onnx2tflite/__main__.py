import argparse
from .converter import onnx_converter

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='onnx model path')
    parser.add_argument('--outpath', type=str, default=None, help='tflite model save path')
    parser.add_argument('--input-node-names', nargs="+", default=None, help='which inputs is you want, support middle layers, None will using onnx orignal inputs')
    parser.add_argument('--output-node-names', nargs="+", default=None, help='which outputs is you want, support middle layers, None will using onnx orignal outputs')
    parser.add_argument('--nosimplify', default=False, action='store_true', help='do not simplify model')
    parser.add_argument("--native-groupconv", default=False, action='store_true', help='using native method for groupconv, only support for tflite version >= 2.9')
    parser.add_argument('--weigthquant', default=False, action='store_true', help='weight only int8 quant')
    parser.add_argument('--fp16', default=False, action='store_true', help='fp16 quant, include input output')
    parser.add_argument('--int8', default=False, action='store_true', help='int8 quant, include input output')
    parser.add_argument('--calibration-data', nargs='+', default=None, help='.npy file paths for INT8 calibration (one per model input, data must be preprocessed)')
    parser.add_argument('--formats', nargs='+', default=['keras', 'tflite'], help='available formats are (h5, tflite)')
    opt = parser.parse_args()
    return opt

def run():
    opt = parse_opt()
    onnx_converter(
        onnx_model_path = opt.weights,
        need_simplify = not opt.nosimplify,
        input_node_names = opt.input_node_names,
        output_node_names = opt.output_node_names,
        output_path = opt.outpath,
        target_formats = opt.formats,
        native_groupconv = opt.native_groupconv,
        weight_quant=opt.weigthquant,
        fp16_model=opt.fp16,
        int8_model=opt.int8,
        calibration_data=opt.calibration_data
    )

if __name__ == "__main__":
    run()