#include "preprocess/agent_preprocess_kernel.cuh"
#include "preprocess/preprocess_kernel.hpp"

#include <cmath>

constexpr int THREADS_PER_BLOCK = 256;

cudaError_t agentPreprocessLauncher(
  const int B, const int N, const int T, const int D, const int C, const int sdc_index,
  const int * target_index, const int * object_type_index, const float * timestamps,
  const float * in_trajectory, float * out_data, bool * out_mask, float * out_last_pos,
  cudaStream_t stream)
{
  dim3 blocks(B, N, T);
  size_t sharedMemSize = sizeof(float) * B * N * T + sizeof(float) * B * N * T * (C + 2) +
                         sizeof(float) * B * N * T * (T + 1) + sizeof(float) * B * N * T * 2;
  agentPreprocessKernel<<<blocks, THREADS_PER_BLOCK, sharedMemSize, stream>>>(
    B, N, T, D, C, sdc_index, target_index, object_type_index, timestamps, in_trajectory, out_data,
    out_mask, out_last_pos);

  return cudaGetLastError();
}

cudaError_t polylinePreprocessLauncher(
  const float * polylines, float * out_polylines, bool * out_mask)
{
  return cudaGetLastError();
}
