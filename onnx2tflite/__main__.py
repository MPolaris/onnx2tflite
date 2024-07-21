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
    parser.add_argument('--imgroot', type=str, default=None, help='when int8=True, imgroot should give for calculating running_mean and running_norm')
    parser.add_argument('--int8mean', type=float, nargs='+', default=[123.675, 116.28, 103.53], help='int8 image preprocesses mean, float or list')
    parser.add_argument('--int8std', type=float, nargs='+', default=[58.395, 57.12, 57.375], help='int8 image preprocesses std, float or list')
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
        int8_mean=opt.int8mean,
        int8_std=opt.int8std,
        image_root=opt.imgroot
    )

if __name__ == "__main__":
    run()