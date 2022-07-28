import os
import onnx
import logging

LOG = logging.getLogger("onnx_loader running:")
try:
    from onnxsim import simplify
except:
    LOG.warning("引入onnxsim.simplify失败")
    def lambda_func(x, *arg, **args):
        return x, False
    simplify = lambda_func

def load_onnx_modelproto(onnx_model_path:str, need_simplify:bool=True):
    if not os.path.exists(onnx_model_path):
        return None
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
            LOG.warning(f"模型优化失败, 从{onnx_model_path}加载")
            model_proto = onnx.load(onnx_model_path)
    return model_proto