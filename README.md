# TensorRT-MTR

> [!WARNING]
> This repository has not been developed yet, and moved in [ktro2828/autoware_mtr](https://github.com/ktro2828/autoware_mtr).

Support of inference with TensorRT for [sshaoshuai/MTR](https://github.com/sshaoshuai/MTR).

## Inputs/Outputs

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

- `scores <float: [B, M]>`
- `trajectory <float: [B, M, Tf, Dt]>`

where,

- `M` ...The number of modes
- `Tf` ...The number of the predicted future frames(=`80`)
- `Dt` ...The number of the predicted trajectory dimensions(=`7`) in the order of `(x, y, dx, dy, yaw, vx, cy)`.

## Build & Run

### Download onnx

```shell
# download onnx.zip
gh release download onnx
```

### Build

```shell
cmake -B build && cmake --build build -j${nproc}
```

### Execute

- With `trtexec`

```shell
# with trtexec
<PATH_TO_TRTEXEC_BIN>/trtexec --onnx=<PATH_TO_ONNX> --staticPlugins=./build/libtrtmtr_plugin.so
```

- With `executable`

Fist, please install `trtmtr` with `cmake --install <DIR>`:

```shell
sudo cmake --install build
```

> [!NOTE]
> Note that, `$LD_LIBRARY_PATH` includes `/usr/local/lib`.  
> If not, append `export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"` to your `.bashrc`.

Then, run the following command:

```shell
trtmtr <PATH_TO_ONNX_OR_ENGINE> [--dynamic --fp16 -n <NUM_REPEAT>]
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

- [x] TensorRT custom plugins
- [x] CUDA kernels
  - [x] pre-process
  - [x] post-process
- [x] Shape inference
  - [x] static shape
  - [x] dynamic shape
- [x] Inference sample
- [ ] Visualization
- [ ] Evaluation
