import os
import onnx
import logging
from onnxsim import simplify

LOG = logging.getLogger("onnx_loader running:")

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
            model_proto, success = simplify(model_proto, check_n=2, dynamic_input_shape=dynamic_input)
        except:
            success = False
        if not success:
            LOG.warning(f"onnxsim is failed, maybe make convert fails.")
            model_proto = onnx.load(onnx_model_path)
    return model_proto