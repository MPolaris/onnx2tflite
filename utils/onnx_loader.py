import os
import onnx
import logging
from onnxsim import simplify

LOG = logging.getLogger("onnx_loader running:")
LOG.setLevel(logging.INFO)

def clean_model_input(model_proto):
    inputs = model_proto.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    names = []
    for initializer in model_proto.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
            names.append(initializer.name)
    LOG.warning(f"[{len(names)}] redundant input nodes are removed.\n \
        nodes name : {','.join(names)}")
    

def load_onnx_modelproto(onnx_model_path:str, need_simplify:bool=True):
    if not os.path.exists(onnx_model_path):
        LOG.error(f"{onnx_model_path} is not exists.")
        raise FileExistsError(f"{onnx_model_path} is not exists.")
    model_proto = onnx.load(onnx_model_path)
    dynamic_input = False
    for inp in model_proto.graph.input:
        for x in inp.type.tensor_type.shape.dim:
            if x.dim_value <= 0:
                dynamic_input = True
                break
    if need_simplify:
        success = False
        try:
            model_proto, success = simplify(model_proto, check_n=1, dynamic_input_shape=dynamic_input)
        except:
            success = False
        if not success:
            LOG.warning(f"onnxsim is failed, maybe make convert fails.")
            model_proto = onnx.load(onnx_model_path)
        clean_model_input(model_proto)
    return model_proto