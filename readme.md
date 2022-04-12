# ONNX转TFLite模型代码
## 注意
- 转换完成使用[comfirm_acc.py](./comfirm_acc.py)确认转换精度。
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
- MLP
- YOLOX
- 部分自定义模型

## 已支持的onnx算子列表
- 卷积Conv
- 分组卷积GroupConv
- 深度可分离卷积DepthwiseConv
- BatchNormalization
- Relu
- Sigmoid
- LeakyRelu
- Tanh
- AveragePool
- MaxPool
- Add
- Sub
- Mul
- Div
- Concat
- Upsample
- Resize
- Constant
- Slice
- Shape
- Gather
- Cast
- Floor
- Unsqueeze
- GlobalAveragePool
- Flatten
- 全连接层Gemm
- Pad
- ReduceMean
- Reciprocal
- Sqrt
- Reshape
- Transpose
- Clip