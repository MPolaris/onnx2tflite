#  ONNX->Keras and ONNX->TFLite tools
## Welcome
If you have some good ideas, welcome to discuss or give project PRs.

## How to use
```cmd
pip install -r requirements.txt
```
```python
# base
python converter.py --weights "./your_model.onnx"

# give save path
python converter.py --weights "./your_model.onnx" --outpath "./save_path"

# save tflite model
python converter.py --weights "./your_model.onnx" --outpath "./save_path" --formats "tflite"

# save keras and tflite model
python converter.py --weights "./your_model.onnx" --outpath "./save_path" --formats "tflite" "keras"

# cutoff model, redefine inputs and outputs, support middle layers
python converter.py --weights "./your_model.onnx" --outpath "./save_path" --formats "tflite" --input-node-names "layer_name" --output-node-names "layer_name1" "layer_name2"

# quantify model weight, only weight
python converter.py --weights "./your_model.onnx" --formats "tflite" --weigthquant

# quantify model weight, include input and output
## recommend
python converter.py --weights "./your_model.onnx" --formats "tflite" --int8 --imgroot "./dataset_path" --int8mean 0 0 0 --int8std 255 255 255
## generate random data, instead of read from image file
python converter.py --weights "./your_model.onnx" --formats "tflite" --int8
```
---
## Features
- High Consistency. Compare to ONNX outputs, average error less than 1e-5 per elements.
- More Faster. Output tensorflow-lite model 30% faster than [onnx_tf](https://github.com/onnx/onnx-tensorflow).
- Auto Channel Align. Auto convert pytorch format(NCWH) to tensorflow format(NWHC).
- Deployment Support. Support output quantitative model, include fp16 quantization and uint8 quantization.
- Code Friendly. I've been trying to keep the code structure simple and clear.
---
## Cautions
- Friendly to 2D vision CNN, and not support 3D CNN, bad support for math operation(such as channel change).
- Please use [comfirm_acc.py](./test/comfirm_acc.py) comfirm output is correct after convertion, because some of methods rely on practice.
- [comfirm_acc.py](./test/comfirm_acc.py) only support tflite, and tflite should not be any quantification.
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
    int8_mean = None,
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
    int8_mean = [123.675, 116.28, 103.53], # give mean of image preprocessing 
    int8_std = [58.395, 57.12, 57.375], # give std of image preprocessing 
    image_root = "./dataset/train" # give image folder of train
)
```
---
## Validated models
- [SSD](https://github.com/qfgaohao/pytorch-ssd)
- [HRNet](HRNet-Facial-Landmark-Detection)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [YOLOV3](https://github.com/ultralytics/yolov3)
- [YOLOV4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
- [YOLOV5](https://github.com/ultralytics/yolov5)
- [YOLOV6](https://github.com/meituan/YOLOv6)
- [YOLOV7](https://github.com/WongKinYiu/yolov7)
- [MoveNet](https://github.com/fire717/movenet.pytorch)
- [UNet\FPN](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets)
- MLP(custom)
- DCGAN(custom)
- [AutoEncoder/VAE](https://github.com/AntixK/PyTorch-VAE)
- all torchvision classification models
- some segmation models in torchvision
- 2D CNN without special operators(custom)
---
## Add operator by yourself
When you counter unspport operator, you can choose add it by yourself or make a issuse.<br/>
It's very simple to implement a new operator parser by following these steps below.<br/>
Step 0: Select a corresponding layer code file in [layers folder](./layers/), such as activations_layers.py for 'HardSigmoid'.<br/>
Step 1: Open it, and edit it:
```python
# all operators regist through OPERATOR register.
# regist operator's name is onnx operator name. 
@OPERATOR.register_operator("HardSigmoid")
class TFHardSigmoid():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        '''
        :param tensor_grap: dict, key is node name, value is tensorflow-keras node output tensor.
        :param node_weights: dict, key is node name, value is static data, such as weight/bias/constant, weight should be transfom by dimension_utils.tensor_NCD_to_NDC_format at most time.
        :param node_inputs: List[str], stored node input names, indicates which nodes the input comes from, tensor_grap and node_weights are possible.
        :param node_attribute: dict, key is attribute name, such as 'axis' or 'perm'. value type is indeterminate, such as List[int] or int or float. notice that type of 'axis' value should be adjusted form NCHW to NHWC by dimension_utils.channel_to_last_dimension or dimension_utils.shape_NCD_to_NDC_format.
        '''
        super().__init__()
        self.alpha = node_attribute.get("alpha", 0.2)
        self.beta = node_attribute.get("beta", 0.5)

    def __call__(self, inputs):
        return tf.clip_by_value(self.alpha*inputs+self.beta, 0, 1)
```
Step 2: Make it work without error.<br/>
Step 3: Convert model to tflite without any quantification.<br/>
Step 4: Run [comfirm_acc.py](./test/comfirm_acc.py), ensure outputs consistency.
## TODO
- [ ] support Transofomer, VIT\Swin Trasnformer etc...
- [x] support cutoff onnx model and specify output layer
- [ ] optimize [comfirm_acc.py](./test/comfirm_acc.py)

## Emmmmmmm
Welcome friendly discuss from any user/person, also for code reference. \
This project is very useful and easy to use for most of regular network, at lease I think.\
It's too disgusting for first(batch) or second(channel) axis change. There are always circumstances that have not been taken into account.

# License
This software is covered by Apache-2.0 license.
