# TensorRT-MTR

```shell
./PATH_TO_TRTEXEC_BIN --onnx=./data/mtr_encoder.onnx \
    --plugins=build/libattention_plugin.so \
    --plugins=build/libknn_plugin.so
```