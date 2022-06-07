#  ONNX->Keras and ONNX->TFLite tools

## How to use
```python
# base
python converter.py --weights "./your_model.onnx"

# give save path
python converter.py --weights "./your_model.onnx" --outpath "./save_path"

# save tflite model
python converter.py --weights "./your_model.onnx" --outpath "./save_path" --formats "tflite"

# save keras and tflite model
python converter.py --weights "./your_model.onnx" --outpath "./save_path" --formats "tflite" "keras"

# quantitative model weight, only weight
python converter.py --weights "./your_model.onnx" --formats "tflite" --weigthquant

# quantitative model weight, include input and output
## recommend
python converter.py --weights "./your_model.onnx" --formats "tflite" --int8 --imgroot "./dataset_path" --int8mean 0 0 0 --int8std 1 1 1
## generate random data, instead of read from image file
python converter.py --weights "./your_model.onnx" --formats "tflite" --int8
```
---
## TODO
- [ ] support Transofomer, VIT\Swin Trasnformer etc...
- [ ] support cutoff onnx model and specify output layer
- [ ] add unit test and optimize [comfirm_acc.py](./test/comfirm_acc.py)
- [ ] add simple code comments.
---

## 注意(Caution)
- please use [comfirm_acc.py](./test/comfirm_acc.py) comfirm output is correct after convertion, because some of methods rely on practice.
- comfirm_acc.py only support tflite, and tflite should not be any quantification.
---

## Pytorch -> ONNX -> Tensorflow-Keras -> Tensorflow-Lite

- ### From torchvision to tensorflow-lite
```python
import torch
import torchvision
_input = torch.randn(1, 3, 224, 224)
model = torchvision.models.mobilenet_v2(True)
# use default settings is ok
torch.onnx.export(model, _input, './mobilenetV2.onnx', opset_version=11)# or opset_version=13

from converter import onnx_converter
onnx_converter(
    onnx_model_path = "./mobilenetV2.onnx",
    need_simplify = True,
    output_path = "./",
    target_formats = ['tflite'], # or ['keras'], ['keras', 'tflite']
    weight_quant = False,
    int8_model = False,
    int8_mean = None
    int8_std = None,
    image_root = None
)
```
- ### From custom pytorch model to tensorflow-lite-int8
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

model = MyModel()
model.load_state_dict(torch.load("model_checkpoint.pth", map_location="cpu"))

_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, _input, './mymodel.onnx', opset_version=11)# or opset_version=13

from converter import onnx_converter
onnx_converter(
    onnx_model_path = "./mymodel.onnx",
    need_simplify = True,
    output_path = "./",
    target_formats = ['tflite'], #or ['keras'], ['keras', 'tflite']
    weight_quant = False,
    int8_model = True, # do quantification
    int8_mean = [0.485, 0.456, 0.406], # give mean of image preprocessing 
    int8_std = [0.229, 0.224, 0.225], # give std of image preprocessing 
    image_root = "./dataset/train" # give image folder of train
)
```

---
## 已验证的模型列表(support models)
- UNet\FPN
- YOLOX
- YOLOV3
- YOLOV4
- YOLOV5
- MobileNetV2 SSD-Lite
- MoveNet
- DCGAN
- MLP
- AutoEncoder/VAE
- normal CNN
- Resnet(torchvision)
- Resnext(torchvision)
- mnasnet(torchvision)
- squeezenet(torchvision)
- Densenet(torchvision)
- shufflenet(torchvision)
- Mobilenet(torchvision)
- Inceptionnet(torchvision)
- Alexnet(torchvision)
- VGG(torchvision)
- DeeplabV3(torchvision)

---
