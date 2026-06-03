#  ONNX->Keras and ONNX->TFLite tools
## Welcome
If you have some good ideas, welcome to discuss or give project PRs.

## Install
```cmd
git clone https://github.com/MPolaris/onnx2tflite.git
cd onnx2tflite
python setup.py install
```
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
## CLI Usage
```cmd
# basic conversion
python -m onnx2tflite --weights "./your_model.onnx"

# save to specific path
python -m onnx2tflite --weights "./your_model.onnx" --outpath "./save_path"

# output formats
python -m onnx2tflite --weights "./your_model.onnx" --formats "tflite"
python -m onnx2tflite --weights "./your_model.onnx" --formats "tflite" "keras"

# subgraph extraction (cutoff model by layer names)
python -m onnx2tflite --weights "./your_model.onnx" --input-node-names "layer_in" --output-node-names "layer_out1" "layer_out2"

# quantization
python -m onnx2tflite --weights "./your_model.onnx" --formats "tflite" --weigthquant     # weight-only int8
python -m onnx2tflite --weights "./your_model.onnx" --formats "tflite" --fp16              # fp16
python -m onnx2tflite --weights "./your_model.onnx" --formats "tflite" --int8              # int8 (random calib, low accuracy)
python -m onnx2tflite --weights "./your_model.onnx" --formats "tflite" --int8              # int8 with calibration
    --calibration-data "./calib_input.npy"

# multi-input model with int8
python -m onnx2tflite --weights "./your_model.onnx" --formats "tflite" --int8
    --calibration-data "./input0.npy" "./input1.npy"

# direct IR conversion (bypass Keras, faster)
python -m onnx2tflite --weights "./your_model.onnx" --formats "tflite" --direct-ir
```
---
## Features
- High Consistency. Compare to ONNX outputs, average error less than 1e-5 per elements.
- More Faster. Output tensorflow-lite model 30% faster than [onnx_tf](https://github.com/onnx/onnx-tensorflow).
- Auto Channel Align. Auto convert pytorch format(NCHW) to tensorflow format(NHWC).
- Deployment Support. FP16, INT8 (weight-only, full integer) quantization.
- Two Backends: Keras (stable) and Direct IR (fast, bypasses Keras).
- Multi-Input INT8: calibration data via .npy files, supports models with multiple inputs.
---
## INT8 Calibration Data Format

INT8 quantization requires a representative dataset for calibration.  
Calibration data is provided as `.npy` files — one file per model input.

**Data format:**
- Shape: `(N, ...)` where N is the number of calibration samples (recommended: 50-200)
- The remaining dimensions must match the model input shape
- Data should be **preprocessed** (normalized, resized, channel-ordered) before saving

**Multi-input models:** provide one `.npy` file per input in the same order as the ONNX model inputs.

```python
import numpy as np

# Single-input model: shape (1, 3, 224, 224)
calib = np.random.randn(100, 3, 224, 224).astype(np.float32)
np.save("./calib_input.npy", calib)

# Multi-input model: two inputs with shapes (1, 3, 224, 224) and (1, 1000)
np.save("./calib_img.npy", np.random.randn(100, 3, 224, 224).astype(np.float32))
np.save("./calib_vec.npy", np.random.randn(100, 1000).astype(np.float32))

onnx_converter(
    onnx_model_path = "./model.onnx",
    int8_model = True,
    calibration_data = ["./calib_img.npy", "./calib_vec.npy"],
)
```
---
## Direct IR Backend

The `use_direct_ir=True` flag enables a new conversion path that builds TFLite
FlatBuffer models directly from the ONNX graph, bypassing Keras entirely.

- Faster conversion (no Keras model building or saving)
- More control over the generated TFLite ops
- Supports most common operators

```python
from onnx2tflite import onnx_converter

onnx_converter("model.onnx", use_direct_ir=True)
```

The Keras backend (`use_direct_ir=False`, default) remains available for operators
not yet supported by the direct IR path (e.g. Erf).
---
## Python API

```python
from onnx2tflite import onnx_converter

# Basic conversion
onnx_converter("model.onnx", output_path="./out", target_formats=['tflite'])

# INT8 quantization with calibration data
onnx_converter(
    "model.onnx",
    int8_model=True,
    calibration_data=["./calib_input.npy"],
)

# Direct IR backend (bypass Keras)
onnx_converter("model.onnx", use_direct_ir=True)
```

### onnx_converter parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| onnx_model_path | str | required | Path to ONNX model file |
| output_path | str | None | Output directory (default: same as input) |
| input_node_names | list | None | Subgraph input names (None = all) |
| output_node_names | list | None | Subgraph output names (None = all) |
| need_simplify | bool | True | Run onnx-simplifier before conversion |
| target_formats | list | ['tflite'] | Output formats: 'tflite', 'keras' |
| use_direct_ir | bool | False | Use direct IR backend (bypass Keras) |
| native_groupconv | bool | False | Use native grouped conv (tflite >= 2.9) |
| weight_quant | bool | False | Weight-only INT8 quantization |
| fp16_model | bool | False | FP16 quantization |
| int8_model | bool | False | Full INT8 quantization |
| calibration_data | list | None | List of .npy file paths for INT8 calibration |
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
When you counter unspported operator, you can choose to add it by yourself or make an issue.

### Keras backend
Register a handler class in `keras_backend/ops/` (e.g. `activation.py`):
```python
@OPERATOR.register_operator("HardSigmoid")
class TFHardSigmoid():
    def __init__(self, tensor_graph, node_weights, node_inputs,
                 node_attribute, node_outputs, layout_dict, *args, **kwargs):
        super().__init__()
        self.alpha = node_attribute.get("alpha", 0.2)
        self.beta = node_attribute.get("beta", 0.5)

    def __call__(self, inputs):
        return tf.clip_by_value(self.alpha*inputs+self.beta, 0, 1)
```

### TFLite direct IR backend
Add a handler function in `tflite_backend/ops/`:
```python
@_register("HardSigmoid")
def _hard_sigmoid(builder, node):
    inp = builder._tensor_map[node.input[0]]
    ...
```
---
# License
This software is covered by Apache-2.0 license.
