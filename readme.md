#  ONNX->Keras and ONNX->TFLite tools
## Welcome
If you have some good ideas, welcome to discuss or give project PRs.

## How to install
```cmd
git clone https://github.com/MPolaris/onnx2tflite.git
cd onnx2tflite
python setup.py install
```
## How to use
```python
from onnx2tflite import onnx_converter
res = onnx_converter(
        onnx_model_path = "./model.onnx",
        need_simplify = True,
        output_path = "./models/",
        target_formats = ['tflite'],
    )
```
---
```cmd
# base
python -m onnx2tflite --weights "./your_model.onnx"

# give save path
python -m onnx2tflite --weights "./your_model.onnx" --outpath "./save_path"

# save tflite model
python -m onnx2tflite --weights "./your_model.onnx" --outpath "./save_path" --formats "tflite"

# save keras and tflite model
python -m onnx2tflite --weights "./your_model.onnx" --outpath "./save_path" --formats "tflite" "keras"

# cutoff model, redefine inputs and outputs, support middle layers
python -m onnx2tflite --weights "./your_model.onnx" --outpath "./save_path" --formats "tflite" --input-node-names "layer_inputname" --output-node-names "layer_outname1" "layer_outname2"

# quantify model weight, only weight
python -m onnx2tflite --weights "./your_model.onnx" --formats "tflite" --weigthquant

# quantify model weight, include input and output
## fp16
python -m onnx2tflite --weights "./your_model.onnx" --formats "tflite" --fp16
## recommend
python -m onnx2tflite --weights "./your_model.onnx" --formats "tflite" --int8 --imgroot "./dataset_path" --int8mean 0 0 0 --int8std 255 255 255
## generate random data, instead of read from image file
python -m onnx2tflite --weights "./your_model.onnx" --formats "tflite" --int8
```
---
## Features
- High Consistency. Compare to ONNX outputs, average error less than 1e-5 per elements.
- More Faster. Output tensorflow-lite model 30% faster than [onnx_tf](https://github.com/onnx/onnx-tensorflow).
- Auto Channel Align. Auto convert pytorch format(NCWH) to tensorflow format(NWHC).
- Deployment Support. Support output quantitative model, include fp16 quantization and uint8 quantization.
- Code Friendly. I've been trying to keep the code structure simple and clear.
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
    fp16_model=False,
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
- [YOLOV10](https://github.com/THU-MIG/yolov10)
- [MoveNet](https://github.com/fire717/movenet.pytorch)
- [UNet\FPN](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets)
- ViT(torchvision)
- [SwinTransformerV1](https://github.com/microsoft/Swin-Transformer)
- MLP(custom)
- DCGAN(custom)
- [AutoEncoder/VAE](https://github.com/AntixK/PyTorch-VAE)
- all torchvision classification models
- some segmation models in torchvision
- 1D or 2D CNN without special operators(custom)
---
## Add operator by yourself
When you counter unspported operator, you can choose to add it by yourself or make an issue.<br/>
It's very simple to implement a new operator parser by following these steps below.<br/>
Step 0: Select a corresponding layer code file in [layers folder](./onnx2tflite/layers/), such as activations_layers.py for 'HardSigmoid'.<br/>
Step 1: Open it, and edit it:
```python
# all operators regist through OPERATOR register.
# regist operator's name is onnx operator name. 
@OPERATOR.register_operator("HardSigmoid")
class TFHardSigmoid():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs) -> None:
        '''
        :param tensor_grap: dict, key is node name, value is tensorflow-keras node output tensor.
        :param node_weights: dict, key is node name, value is static data, such as weight/bias/constant, weight should be transfom by dimension_utils.tensor_NCD_to_NDC_format at most time.
        :param node_inputs: List[str], stored node input names, indicates which nodes the input comes from, tensor_grap and node_weights are possible.
        :param node_attribute: dict, key is attribute name, such as 'axis' or 'perm'. value type is indeterminate, such as List[int] or int or float. notice that type of 'axis' value should be adjusted form NCHW to NHWC by dimension_utils.channel_to_last_dimension or dimension_utils.shape_NCD_to_NDC_format.
        :param node_inputs: List[str], stored node output names.
        :param layout_dict: List[Layout], stored all before node's layout.
        '''
        super().__init__()
        self.alpha = node_attribute.get("alpha", 0.2)
        self.beta = node_attribute.get("beta", 0.5)

    def __call__(self, inputs):
        return tf.clip_by_value(self.alpha*inputs+self.beta, 0, 1)
```
Step 2: Make it work without error.<br/>
Step 3: Convert model to tflite without any quantification.<br/>

---
## Limitation
- The number of operators can not cover all models.
- Friendly to 1D/2D vision CNN, and not support 3D CNN.
- Model accuracy check may not be accurate.
---
## TODO
- [x] support Transofomer, VIT\Swin Trasnformer etc...
- [x] support cutoff onnx model and specify output layer
- [x] optimize comfirm_acc.py(removed, The output checker will run automatically.)
---
## Emmmmmmm
Finally, I did.

# License
This software is covered by Apache-2.0 license.
