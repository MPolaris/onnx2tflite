# ONNX转TFLite模型代码
## 注意
- 因为大部分是靠实践经验的，所以转换完成最好使用[comfirm_acc.py](./test/comfirm_acc.py)确认转换精度。
- 目前还没有更新完，我需要一点时间。。。
- 目前只支持2D卷积网络
---

## 添加新算子时，API查询地址
- onnx_api : https://github.com/onnx/onnx/blob/main/docs/Operators.md
- tensorflow_api : https://tensorflow.google.cn/api_docs/python/tf
- keras_api : https://keras.io/search.html
---

## 已验证的模型列表
- Resnet
- Densenet
- Mobilenet
- Alexnet
- VGG
- UNet\FPN
- YOLOX
- YOLOV5