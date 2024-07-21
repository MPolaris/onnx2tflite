import os
import pytest

import torch
import torch.nn as nn
from onnx2tflite import onnx_converter

MODEL_ROOT = "./unit_test"
os.makedirs(MODEL_ROOT, exist_ok=True)

@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_squeeze():
    class Squeeze(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

        def forward(self, x):
            x = torch.unsqueeze(x, dim=1)
            # x = torch.tile(x, dims=(2,1,1))
            x = torch.squeeze(x, dim=1)
            
            return x
        
    model = Squeeze()
    x = torch.randn(1,1,1,2)
    
    onnx_model_path = os.path.join(MODEL_ROOT, "test_squeeze.onnx")
    torch.onnx.export(model, x, onnx_model_path, opset_version=11)

    res = onnx_converter(
            onnx_model_path = onnx_model_path,
            need_simplify = True,
            output_path = MODEL_ROOT,
            target_formats = ['tflite'],
            native_groupconv=False,
            fp16_model=False,
            int8_model=False,
        )

    assert res['tflite_error'] < 1e-3