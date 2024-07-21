import os
import logging
from .components import load_onnx_modelproto, keras_builder, tflite_builder, get_elements_error

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("converter running:")

def onnx_converter(onnx_model_path:str,  output_path:str=None, 
                    input_node_names:list=None, output_node_names:list=None,
                    need_simplify:bool=True, target_formats:list = ['keras', 'tflite'],
                    native_groupconv:bool=False,
                    weight_quant:bool=False, fp16_model:bool=False, int8_model:bool=False, image_root:str=None,
                    int8_mean:list or float = [123.675, 116.28, 103.53], int8_std:list or float = [58.395, 57.12, 57.375])->float:
    """
    Converts an ONNX model to various target formats with optional optimizations.

    Parameters:
    onnx_model_path (str): Path to the input ONNX model file.
    output_path (str, optional): Path to save the converted model(s). If None, the converted model(s) will be saved in the same directory as the input model.
    input_node_names (list, optional): List of input node names. If None, the default input nodes of the ONNX model are used.
    output_node_names (list, optional): List of output node names. If None, the default output nodes of the ONNX model are used.
    need_simplify (bool, optional): If True, the ONNX model will be simplified before conversion. Default is True.
    target_formats (list, optional): List of target formats to convert the ONNX model to. Default is ['keras', 'tflite'].
    native_groupconv (bool, optional): If True, retains native group convolution operations during conversion. Default is False.
    weight_quant (bool, optional): If True, applies weight quantization to the converted model. Default is False.
    fp16_model (bool, optional): If True, converts the model to use FP16 precision. Default is False.
    int8_model (bool, optional): If True, converts the model to use INT8 precision. Default is False.
    image_root (str, optional): Path to the root directory of images for calibration if INT8 quantization is enabled. Default is None.
    int8_mean (list or float, optional): Mean values for INT8 quantization. Default is [123.675, 116.28, 103.53].
    int8_std (list or float, optional): Standard deviation values for INT8 quantization. Default is [58.395, 57.12, 57.375].

    Returns:
    float: Error value.

    Note:
    - The function supports multiple target formats for conversion and allows for various optimizations such as simplification, quantization, and precision reduction.
    - When INT8 quantization is enabled, 'image_root', 'int8_mean', and 'int8_std' parameters are used for calibration.
    """
    if not isinstance(target_formats, list) and  'keras' not in target_formats and 'tflite' not in target_formats:
        raise KeyError("'keras' or 'tflite' should in list")
    
    model_proto = load_onnx_modelproto(onnx_model_path, input_node_names, output_node_names, need_simplify)

    keras_model, input_layout, output_layout = keras_builder(model_proto, native_groupconv)

    if 'tflite' in target_formats:
        tflite_model = tflite_builder(keras_model, weight_quant, fp16_model, int8_model, image_root, int8_mean, int8_std)

    onnx_path, model_name = os.path.split(onnx_model_path)
    if output_path is None:
        output_path = onnx_path
    output_path = os.path.join(output_path, model_name.split('.')[0])

    if fp16_model:
        output_path = output_path + "_fp16"
    elif int8_model:
        output_path = output_path + "_int8"

    keras_model_path = None
    if 'keras' in target_formats:
        keras_model_path = output_path + ".h5"
        keras_model.save(keras_model_path)
        LOG.info(f"keras model saved in {keras_model_path}")

    tflite_model_path = None
    if 'tflite' in target_formats:
        tflite_model_path = output_path + ".tflite"
        with open(tflite_model_path, "wb") as fp:
            fp.write(tflite_model)

    convert_result = {"keras":keras_model_path, "tflite":tflite_model_path, "keras_error":0, "tflite_error":0}
    # ignore quantization model
    if int8_model:
        return convert_result
    
    error_dict = {}
    try:
        error_dict = get_elements_error(model_proto, keras_model_path, tflite_model_path, input_layout, output_layout)
        keras_error, tflite_error = error_dict.get("keras", None), error_dict.get("tflite", None)
        if keras_error:
            if keras_error > 1e-2:
                LOG.error("h5 model elements' max error has reached {:^.4E}, but convert is done, please check {} carefully!".format(keras_error, keras_model_path))
            elif keras_error > 1e-4:
                LOG.warning("h5 model elements' max error is {:^.4E}, pass, h5 saved in {}".format(keras_error, keras_model_path))
            else:
                LOG.info("h5 model elements' max error is {:^.4E}, pass, h5 saved in {}".format(keras_error, keras_model_path))
        if tflite_error:
            if tflite_error > 1e-2:
                LOG.error("tflite model elements' max error has reached {:^.4E}, but convert is done, please check {} carefully!".format(tflite_error, tflite_model_path))
            elif tflite_error > 1e-4:
                LOG.warning("tflite model elements' max error is {:^.4E}, pass, tflite saved in {}".format(tflite_error, tflite_model_path))
            else:
                LOG.info("tflite model elements' max error is {:^.4E}, pass, tflite saved in {}".format(tflite_error, tflite_model_path))
    except:
        LOG.warning("convert is successed, but model running is failed, please check carefully!")
    
    convert_result["keras_error"] = error_dict.get("keras", None)
    convert_result["tflite_error"] = error_dict.get("tflite", None)
    return convert_result