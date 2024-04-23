# TensorRT-MTR

Support of inference with TensorRT for [sshaoshuai/MTR](https://github.com/sshaoshuai/MTR).

## Network

### Inputs

- `trajectory <float; [B, N, Tp, Da]>`
- `trajectory_mask <bool: [B, N, Tp]>`
- `polyline <float: [B, K, P, Dp]>`
- `polyline_mask <bool: [B, K, P]>`
- `polyline_center <float: [B, K, 3]>`
- `last_pos <float; [B, N, 3]`
- `track_index <int: [B]>`
- `label_index <int: [N]>`
- `intention_points <float: [B, 64, 3]>`

where,

- `B` ...The number of target agents
- `N` ...The number of all agents
- `Tp` ...The number of past frames(=`11`)
- `Da` ...The number of agent state dimensions(=`29`)
- `K` ...The max number of polylines(=`768`)
- `P` ...The max number of points contained in each polyline(=`20`)
- `Dp` ...The number of polyline state dimensions(=`9`)

### Outputs

- `scores <float: [N, M]>`
- `trajectory <float: [N, M, Tf, Dt]>`

where,

- `M` ...The number of modes
- `Tf` ...The number of the predicted future frames(=`80`)
- `Dt` ...The number of the predicted trajectory dimensions(=`7`)

## Build & Run

### Build

```shell
cmake -B build
cmake --build build -j${nproc}
```

### Download onnx

```shell
gh release download onnx
```

### Execute

```shell
# with trtexec
<PATH_TO_TRTEXEC_BIN>/trtexec --onnx=<PATH_TO_ONNX> --staticPlugins=./build/libcustom_plugin.so

# main
./build/main <PATH_TO_ONNX_OR_ENGINE>
```

### Unittest

```shell
# test agent data container defined in `include/mtr/agent.hpp`
./build/test_agent

# test polyline data container defined in `include/mtr/polyline.hpp`
./build/test_polyline

# test intention point data container defined in `include/mtr/intention_point.hpp`
./build/test_intention_point
```

## TODO

- [x] Add TensorRT custom plugins
- [x] Add pre-process
- [x] Add post-process
- [x] Add inference sample
- [ ] Add visualization
- [ ] Add evaluation
