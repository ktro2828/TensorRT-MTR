#include "postprocess/postprocess_kernel.cuh"

cudaError_t postprocessLauncher(
  const int B, const int M, const int inTime, const int inDim, const int outTime, const int outDim,
  const float * target_state, float * pred_score, float * pred_trajectory, cudaStream_t stream)
{
  return cudaGetLastError();
}