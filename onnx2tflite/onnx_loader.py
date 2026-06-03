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
    
    if len(names) > 0:
        LOG.warning(f"[{len(names)}] redundant input nodes are removed.\n \
            nodes name : {','.join(names)}")

def get_onnx_submodel(onnx_model_path:str, input_node_names:list=None, output_node_names:list=None):
    '''
        cutoff onnx model
    '''
    model_proto = onnx.load(onnx_model_path)
    if input_node_names is None:
        input_node_names = []
        for inp in model_proto.graph.input:
            input_node_names.append(inp.name)

    if output_node_names is None:
        output_node_names = []
        for oup in model_proto.graph.output:
            output_node_names.append(oup.name)
    del model_proto

    new_model_path = os.path.splitext(onnx_model_path)[0] + "_sub.onnx"
    onnx.utils.extract_model(onnx_model_path, new_model_path, input_node_names, output_node_names)
    model_proto = onnx.load(new_model_path)
    return model_proto

def get_proto(onnx_model_path:str, input_node_names:list=None, output_node_names:list=None):
    if input_node_names is None and output_node_names is None:
        return onnx.load(onnx_model_path)
    else:
        return get_onnx_submodel(onnx_model_path, input_node_names, output_node_names)
    
def load_onnx_modelproto(onnx_model_path:str, input_node_names:list=None, output_node_names:list=None, need_simplify:bool=True):
    if not os.path.exists(onnx_model_path):
        LOG.error(f"{onnx_model_path} is not exists.")
        raise FileExistsError(f"{onnx_model_path} is not exists.")
    model_proto = get_proto(onnx_model_path, input_node_names, output_node_names)
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