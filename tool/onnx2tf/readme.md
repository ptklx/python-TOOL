# ONNX转TFLite模型代码
## 注意事项
- 最好使用onnxsim优化过的ONNX模型, 目前已经集成onnxsim。
- 遇到不支持的算子或代码错误, 自行添加。
- 转换完成使用[comfirm_acc.py](./comfirm_acc.py)确认转换精度。
---

## 添加新算子时，API查询地址
- onnx_api : https://github.com/onnx/onnx/blob/main/docs/Operators.md
- tensorflow_api : https://tensorflow.google.cn/api_docs/python/tf
- keras_api : https://keras.io/search.html
---

## 已验证的模型列表
- Resnet
- Densenet
- Inceptionnet
- Mobilenet
- Alexnet
- VGG
- UNet\FPN
- YOLOX
- YOLOV3
- YOLOV4
- YOLOV5
- MobileNetV2 SSD-Lite
- MoveNet
- BigGAN
- DCGAN
- 部分自定义模型

## 已支持的onnx算子列表
- 卷积Conv
- 分组卷积GroupConv
- 分离卷积DepthwiseConv
- BatchNormalization
- Relu
- Sigmoid
- LeakyRelu
- Tanh
- Softmax
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
- Split
- ScatterND
- Shape
- Gather
- Cast
- Floor
- Squeeze
- Unsqueeze
- GlobalAveragePool
- Flatten
- 全连接层Gemm
- Pad
- ReduceMean
- Reciprocal
- Sqrt
- Split
- Reshape
- Transpose
- Clip
- GatherElements
- ArgMax
- MatMul
- Selu