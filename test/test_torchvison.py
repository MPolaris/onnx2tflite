'''
    unit test for torchvision models
'''
import os
import pytest

import torch
import torchvision
from onnx2tflite import onnx_converter

MODEL_ROOT = "./unit_test"
os.makedirs(MODEL_ROOT, exist_ok=True)

@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_resnet():
    model = torchvision.models.resnet18(False)
    onnx_model_path = os.path.join(MODEL_ROOT, "resnet18.onnx")
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), onnx_model_path, opset_version=13)
    error = onnx_converter(onnx_model_path, need_simplify = True, output_path = MODEL_ROOT, target_formats = ['tflite'])['tflite_error']
    assert error < 1e-3

@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_mobilenet():
    model = torchvision.models.mobilenet_v2(False)
    onnx_model_path = os.path.join(MODEL_ROOT, "mobilenet_v2.onnx")
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), onnx_model_path, opset_version=13)
    error = onnx_converter(onnx_model_path, need_simplify = True, output_path = MODEL_ROOT, target_formats = ['tflite'])['tflite_error']
    assert error < 1e-3

@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_deeplabv3():
    model = torchvision.models.segmentation.deeplabv3_resnet50(False)
    onnx_model_path = os.path.join(MODEL_ROOT, "deeplabv3_resnet50.onnx")
    torch.onnx.export(model, torch.randn(1, 3, 512, 1024), onnx_model_path, opset_version=13)
    error = onnx_converter(onnx_model_path, need_simplify = True, output_path = MODEL_ROOT, target_formats = ['tflite'])['tflite_error']
    assert error < 1e-3

@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_vit():
    model = torchvision.models.vit_b_16(False)
    onnx_model_path = os.path.join(MODEL_ROOT, "vit_b_16.onnx")
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), onnx_model_path, opset_version=13)
    error = onnx_converter(onnx_model_path, need_simplify = True, output_path = MODEL_ROOT, target_formats = ['tflite'])['tflite_error']
    assert error < 1e-3