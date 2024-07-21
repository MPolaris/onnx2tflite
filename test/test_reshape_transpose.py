import os
import pytest

import torch
import torch.nn as nn
from onnx2tflite import onnx_converter

MODEL_ROOT = "./unit_test"
os.makedirs(MODEL_ROOT, exist_ok=True)

@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_reshape_trans():
    class test1(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.conv1 = nn.Conv2d(3, 3, 3, 2, 1)
            self.conv2 = nn.Conv2d(3, 3, 3, 2, 1)

        def forward(self, x):
            x = torch.reshape(x, (1, 3, 32, 16))
            # x = torch.transpose(x, (0, 1, 3, 2))
            x = torch.transpose(x, 3, 2)
            x = self.conv1(x)
            x = self.conv2(x)
            return x
        
    model = test1()
    x = torch.randn(1, 3*32*16)

    onnx_model_path = os.path.join(MODEL_ROOT, "test_reshape_trans.onnx")
    torch.onnx.export(model, x, onnx_model_path, opset_version=11)

    res = onnx_converter(
            onnx_model_path = onnx_model_path,
            need_simplify = True,
            output_path = MODEL_ROOT,
            target_formats = ['tflite'],
            native_groupconv=False,
            fp16_model=False,
            int8_model = False,
        )

    assert res['tflite_error'] < 1e-3