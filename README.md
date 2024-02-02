# TensorRT-MTR

Support of inference with TensorRT for [sshaoshuai/MTR](https://github.com/sshaoshuai/MTR).

## Network

### Inputs
- `trajectory <float; [N, A, Tp, Da]>`
- `trajectory_mask <bool; [N, A, Tp]>`
- `polyline <float; [N, K, P, Dp]>`
- `polyline_mask <bool; [N, K, P]>`
- `polyline_center <float; [N, K, 3]>`
- `last_pos <float; [N, A, 3]`
- `track_index <int; [N]>`
- `label_index <int; [N]>`

where, 

- `N` ...The number of targets
- `A` ...The number of agents
- `Tp` ...The number of past frames(=`11`)
- `Da` ...The number of agent tensor dimensions(=`29`)
- `K` ...The max number of polylines(=`768`)
- `P` ...The max number of points contained in each polyline(=`20`)
- `Dp` ...The number of polyline tensor dimensions(=`9`)

### Outputs
- `scores <float; [N, M]>`
- `trajectory <float; [N, M, Tf, Dt]>`

where,

- `M` ...The number of modes
- `Tf` ...The number of the predicted future frames(=`80`)
- `Dt` ...The number of the predicted trajectory dimensions(=`7`)

## Build & Run

```shell
cmake -B build
cmake --build build -j${nproc}
```

```shell
# with trtexec
<PATH_TO_TRTEXEC_BIN>/trtexec --onnx=<PATH_TO_ONNX> \
    --plugins=./build/libattention_plugin.so \
    --plugins=./build/libknn_plugin.so

# main
./build/main <PATH_TO_ONNX_OR_ENGINE>
```

## TODO

- [x] Add TensorRT custom plugins
- [ ] Add inference sample