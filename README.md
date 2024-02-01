# TensorRT-MTR

Support of inference with TensorRT for [sshaoshuai/MTR](https://github.com/sshaoshuai/MTR).

## Run

```shell
cd <PATH_TO_TRTEXEC_BIN>
./trtexec --onnx=<PATH_TO_WORKSPACE>/data/mtr_static.onnx \
    --plugins=<PATH_TO_WORKSPACE>/build/libattention_plugin.so \
    --plugins=<PATH_TO_WORKSPACE>build/libknn_plugin.so
```

## TODO

- [x] Add TensorRT custom plugins
- [ ] Add inference sample