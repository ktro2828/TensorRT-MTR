#include "preprocess/polyline_preprocess_kernel.cuh"

#include <iostream>

__global__ void transformPolylineKernel(
  const int K, const int P, const int PointDim, const float * in_polyline, const int B,
  const int AgentDim, const float * target_state, float * out_polyline, bool * out_polyline_mask)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int p = blockIdx.z * blockDim.z + threadIdx.z;

  if (b >= B || k >= K || p >= P) {
    return;
  }

  const int src_polyline_idx = (k * P + p) * PointDim;
  const float x = in_polyline[src_polyline_idx];
  const float y = in_polyline[src_polyline_idx + 1];
  const float z = in_polyline[src_polyline_idx + 2];
  const float dx = in_polyline[src_polyline_idx + 3];
  const float dy = in_polyline[src_polyline_idx + 4];
  const float dz = in_polyline[src_polyline_idx + 5];
  const float type_id = in_polyline[src_polyline_idx + 6];

  const int center_idx = b * AgentDim;
  const float center_x = target_state[center_idx];
  const float center_y = target_state[center_idx + 1];
  const float center_z = target_state[center_idx + 2];
  const float center_yaw = target_state[center_idx + 6];
  const float center_cos = cos(center_yaw);
  const float center_sin = sin(center_yaw);

  // do transform
  const float trans_x = center_cos * (x - center_x) - center_sin * (y - center_y);
  const float trans_y = center_sin * (x - center_x) + center_cos * (y - center_y);
  const float trans_z = z - center_z;
  const float trans_dx = center_cos * dx - center_sin * dy;
  const float trans_dy = center_sin * dx + center_cos * dy;
  const float trans_dz = dz;

  const int out_idx = (b * K * P + k * P + p) * (PointDim + 2);
  out_polyline[out_idx] = trans_x;
  out_polyline[out_idx + 1] = trans_y;
  out_polyline[out_idx + 2] = trans_z;
  out_polyline[out_idx + 3] = trans_dx;
  out_polyline[out_idx + 4] = trans_dy;
  out_polyline[out_idx + 5] = trans_dz;
  out_polyline[out_idx + 6] = type_id;

  const int out_mask_idx = b * K * P + k * P + p;
  bool is_valid = false;
  for (size_t i = 0; i < 6; ++i) {
    is_valid += out_polyline[out_idx + i] != 0.0f;
  }
  out_polyline_mask[out_mask_idx] = is_valid;
}

__global__ void setPreviousPositionKernel(
  const int B, const int K, const int P, const int D, const bool * mask, float * polyline)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int p = blockIdx.z * blockDim.z + threadIdx.z;

  if (b >= B || k >= K || p >= P) {
    return;
  }

  const int cur_idx = (b * K * P + k * P + p) * D;
  const int pre_idx = k == 0 ? cur_idx : (b * K * P + (k - 1) * P + p) * D;

  polyline[cur_idx + D - 2] = polyline[pre_idx];
  polyline[cur_idx + D - 1] = polyline[pre_idx + 1];

  const int mask_idx = b * K * P + k * P + p;
  if (!mask[mask_idx]) {
    for (int d = 0; d < D; ++d) {
      polyline[cur_idx + d] = 0.0f;
    }
  }
}

__global__ void calculateTopkKernel(
  const int K, const int L, const int B, const float * distances, int * topk_index)
{
  /* TODO */
}

__global__ void calculatePolylineCenterKernel(
  const int B, const int K, const int P, const int PointDim, const float * polyline,
  const float * polyline_mask, float * polyline_center)
{
  /* TODO */
  // p = threadIdx.x
  // extern __shared__ float shared_memory[];
  // float * shared_center = shared_memory;                 // [B*K*3]
  // float * shared_num_valid = &shared_center[B * K * 3];  // [B*K]
}

cudaError_t polylinePreprocessWithTopkLauncher(
  const int L, const int K, const int P, const int PointDim, const float * in_polyline, const int B,
  const int AgentDim, const float * target_state, int * topk_index, float * out_polyline,
  bool * out_polyline_mask, float * out_polyline_center, cudaStream_t stream)
{
  if (L < K) {
    std::cerr << "L must be greater than K, but got L: " << L << ", K: " << K << std::endl;
    return cudaError_t::cudaErrorInvalidValue;
  }

  return cudaGetLastError();
}

cudaError_t polylinePreprocessLauncher(
  const int K, const int P, const int PointDim, const float * in_polyline, const int B,
  const int AgentDim, const float * target_state, float * out_polyline, bool * out_polyline_mask,
  float * out_polyline_center, cudaStream_t stream)
{
  // TODO(ktro2828): update the number of blocks and threads to guard `cudaErrorIllegalAccess: an
  // illegal memory access was encounted.`
  constexpr int threadsPerBlock = 256;
  const dim3 blocks(
    (B - threadsPerBlock + 1) / threadsPerBlock, (K - threadsPerBlock + 1) / threadsPerBlock,
    (P - threadsPerBlock + 1) / threadsPerBlock);

  transformPolylineKernel<<<blocks, threadsPerBlock, 0, stream>>>(
    K, P, PointDim, in_polyline, B, AgentDim, target_state, out_polyline, out_polyline_mask);

  setPreviousPositionKernel<<<blocks, threadsPerBlock, 0, stream>>>(
    B, K, P, PointDim, out_polyline_mask, out_polyline);

  return cudaGetLastError();
}