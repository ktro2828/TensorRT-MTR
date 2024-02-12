#include "mtr/mtr.hpp"

#include "preprocess/preprocess_kernel.hpp"

namespace mtr
{
TrtMTR::TrtMTR(
  const std::string & model_path, const std::string & precision,
  const std::vector<std::string> target_labels, const BatchConfig & batch_config,
  const size_t max_workspace_size, const BuildConfig & build_config)
: target_labels_(target_labels)
{
  builder_ = std::make_unique<MTRBuilder>(
    model_path, precision, batch_config, max_workspace_size, build_config);
  builder_->setup();

  if (!builder_->isInitialized()) {
    return;
  }
}

bool TrtMTR::doInference()
{
  return preProcess();
}

bool TrtMTR::preProcess()
{
  constexpr int B = 2;
  constexpr int N = 4;
  constexpr int T = 5;
  constexpr int D = 10;
  constexpr int C = 3;
  constexpr int sdc_index = 1;

  int h_target_index[B] = {0, 2};
  int h_object_type_index[N] = {0, 0, 2, 1};
  float h_timestamps[T] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float h_trajectory[N][T][D] = {
    {{1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 1.0f},
     {10.0f, 20.0f, 1.0f, 0.1f, 0.2f, 1.0f, 0.5f, 3.0f, 0.1f, 1.0f}},
    {{2.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {30.0f, 40.0f, 2.0f, 0.1f, 0.2f, 1.0f, 0.25f, 3.0f, 0.1f, 0.0f}},
    {{2.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {3.0f, 3.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f}},
    {{2.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {3.0f, 3.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f}}};

  int *d_target_index, *d_object_type_index;
  float *d_timestamps, *d_trajectory;
  // allocate input memory
  cudaMalloc(reinterpret_cast<void **>(&d_target_index), sizeof(int) * B);
  cudaMalloc(reinterpret_cast<void **>(&d_object_type_index), sizeof(int) * N);
  cudaMalloc(reinterpret_cast<void **>(&d_timestamps), sizeof(float) * T);
  cudaMalloc(reinterpret_cast<void **>(&d_trajectory), sizeof(float) * N * T * D);
  // copy input data
  cudaMemcpy(d_target_index, h_target_index, sizeof(int) * B, cudaMemcpyHostToDevice);
  cudaMemcpy(d_object_type_index, h_object_type_index, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_timestamps, h_timestamps, sizeof(float) * T, cudaMemcpyHostToDevice);
  cudaMemcpy(d_trajectory, h_trajectory, sizeof(float) * N * T * D, cudaMemcpyHostToDevice);

  float *d_out_data, *d_out_last_pos;
  bool * d_out_mask;
  size_t outDataSize = sizeof(float) * B * N * T * (D - 2 + C + 2 + T + 1);
  size_t outMaskSize = sizeof(bool) * B * N * T;
  size_t outLastPosSize = sizeof(float) * B * N * 3;
  // allocate output memory
  cudaMalloc(reinterpret_cast<void **>(&d_out_data), outDataSize);
  cudaMalloc(reinterpret_cast<void **>(&d_out_mask), outMaskSize);
  cudaMalloc(reinterpret_cast<void **>(&d_out_last_pos), outLastPosSize);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // DEBUG
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaEventQuery(start);
  // Preprocess
  auto error_code = agentPreprocessLauncher(
    B, N, T, D, C, sdc_index, d_target_index, d_object_type_index, d_timestamps, d_trajectory,
    d_out_data, d_out_mask, d_out_last_pos, stream);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("Processing time = %g ms.\n", elapsed_time);

  if (error_code != cudaSuccess) {
    std::cerr << "ERROR: " << cudaGetErrorString(error_code) << std::endl;
    return false;
  } else {
    std::cout << "SUCCESS: success to inference" << std::endl;
    return true;
  }
}
}  // namespace mtr