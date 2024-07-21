import os
import pytest

import torch
import torch.nn as nn
from onnx2tflite import onnx_converter

MODEL_ROOT = "./unit_test"
os.makedirs(MODEL_ROOT, exist_ok=True)

@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_concat():
    class Concat(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.conv1 = nn.Conv2d(3, 3, 3, 2, 1)
            # self.conv2 = nn.Conv2d(3, 3, 3, 2, 1)
            self._const = torch.randn(1,2,16,8)

        def forward(self, x1, x2, x3):
            x1 = torch.reshape(x1, (1, 3, 16, 8))
            # x = torch.transpose(x, (0, 1, 3, 2))
            x2 = torch.transpose(x2, 3, 2)
            x3 = self.conv1(x3)
            x = torch.concat([x1,x2,x3,self._const], dim=1)
            return x
        
    model = Concat()
    x1 = torch.randn(1,3*16*8)
    x2 = torch.randn(1,3,8,16)
    x3 = torch.randn(1,3,32,16)

    onnx_model_path = os.path.join(MODEL_ROOT, "test_concat.onnx")
    torch.onnx.export(model, (x1,x2,x3), onnx_model_path, opset_version=11)

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