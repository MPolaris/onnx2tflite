import os
import onnx
import numpy as np
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
    if need_simplify:
        model_proto, success = simplify(onnx_model_path, check_n=2)
        if not success:
            LOG.warning(f"模型优化失败, 从{onnx_model_path}加载")
            model_proto = onnx.load(onnx_model_path)
    else:
        model_proto = onnx.load(onnx_model_path)
    return model_proto