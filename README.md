# TensorRT-MTR

```shell
cd <PATH_TO_TRTEXEC_BIN>
./trtexec --onnx=<PATH_TO_WORKSPACE>/data/mtr_static.onnx \
    --plugins=<PATH_TO_WORKSPACE>/build/libattention_plugin.so \
    --plugins=<PATH_TO_WORKSPACE>build/libknn_plugin.so
```