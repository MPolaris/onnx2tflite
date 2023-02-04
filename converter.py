import os
import logging
import argparse
from utils import load_onnx_modelproto, keras_builder, tflite_builder, get_elements_error
__version__ = __VERSION__ = "1.0.0"

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("converter running:")

def onnx_converter(onnx_model_path:str,  output_path:str=None, 
                    input_node_names:list=None, output_node_names:list=None,
                    need_simplify:bool=True, target_formats:list = ['keras', 'tflite'],
                    native_groupconv:bool=False,
                    weight_quant:bool=False, int8_model:bool=False, image_root:str=None,
                    int8_mean:list or float = [123.675, 116.28, 103.53], int8_std:list or float = [58.395, 57.12, 57.375])->float:
    if not isinstance(target_formats, list) and  'keras' not in target_formats and 'tflite' not in target_formats:
        raise KeyError("'keras' or 'tflite' should in list")
    
    model_proto = load_onnx_modelproto(onnx_model_path, input_node_names, output_node_names, need_simplify)

    keras_model = keras_builder(model_proto, native_groupconv)

    if 'tflite' in target_formats:
        tflite_model = tflite_builder(keras_model, weight_quant, int8_model, image_root, int8_mean, int8_std)

    onnx_path, model_name = os.path.split(onnx_model_path)
    if output_path is None:
        output_path = onnx_path
    output_path = os.path.join(output_path, model_name.split('.')[0])

    max_error = 1000
    if 'keras' in target_formats:
        keras_model.save(output_path + ".h5")
        LOG.info(f"keras model saved in {output_path}.h5")

    if 'tflite' in target_formats:
        tflite_model_path = output_path + ".tflite"
        with open(tflite_model_path, "wb") as fp:
            fp.write(tflite_model)
        try:
            max_error = get_elements_error(model_proto, tflite_model_path)
            if max_error > 1e-2:
                LOG.error("elements' max error has reached {:^.4f}, but convert is done, please check {} carefully!".format(max_error, tflite_model_path))
            elif max_error > 1e-4:
                LOG.warning("elements' max error is {:^.5f}, pass, tflite saved in {}".format(max_error, tflite_model_path))
            else:
                LOG.info("elements' max error is {:^.6f}, pass, tflite saved in {}".format(max_error, tflite_model_path))
        except:
            LOG.warning("convert is successed, but model running is failed, please check {} carefully!".format(tflite_model_path))
    return max_error

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='onnx model path')
    parser.add_argument('--outpath', type=str, default=None, help='tflite model save path')
    parser.add_argument('--input-node-names', nargs="+", default=None, help='which inputs is you want, support middle layers, None will using onnx orignal inputs')
    parser.add_argument('--output-node-names', nargs="+", default=None, help='which outputs is you want, support middle layers, None will using onnx orignal outputs')
    parser.add_argument('--nosimplify', default=False, action='store_true', help='do not simplify model')
    parser.add_argument("--native-groupconv", default=False, action='store_true', help='using native method for groupconv, only support for tflite version >= 2.9')
    parser.add_argument('--weigthquant', default=False, action='store_true', help='tflite weigth int8 quant')
    parser.add_argument('--int8', default=False, action='store_true', help='tflite weigth int8 quant, include input output')
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
        int8_model=opt.int8,
        int8_mean=opt.int8mean,
        int8_std=opt.int8std,
        image_root=opt.imgroot
    )

if __name__ == "__main__":
    run()