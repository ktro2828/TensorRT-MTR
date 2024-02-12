#include "preprocess/agent_preprocess_kernel.cuh"

#include <stdio.h>

__device__ void transform_trajectory(
  const int B, const int N, const int T, const int D, const int * target_index,
  const float * in_trajectory, float * output)
{
  // output [B * N * T * D]
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;

  if (b < B && n < N && t < T) {
    int src_idx = n * T + t;
    const float x = in_trajectory[src_idx * D];
    const float y = in_trajectory[src_idx * D + 1];
    const float z = in_trajectory[src_idx * D + 2];
    const float dx = in_trajectory[src_idx * D + 3];
    const float dy = in_trajectory[src_idx * D + 4];
    const float dz = in_trajectory[src_idx * D + 5];
    const float yaw = in_trajectory[src_idx * D + 6];
    const float vx = in_trajectory[src_idx * D + 7];
    const float vy = in_trajectory[src_idx * D + 8];
    const float is_valid = in_trajectory[src_idx * D + 9];

    // transform for each target
    const int center_idx = (target_index[b] * T + T - 1) * D;
    const float center_x = in_trajectory[center_idx];
    const float center_y = in_trajectory[center_idx + 1];
    const float center_yaw = in_trajectory[center_idx + 6];
    const float cos_val = std::cos(center_yaw);
    const float sin_val = std::sin(center_yaw);

    // transform
    const float trans_x = cos_val * x - sin_val * y - center_x;
    const float trans_y = sin_val * x + cos_val * y - center_y;
    const float trans_vx = cos_val * vx - sin_val * vy;
    const float trans_vy = sin_val * vx + cos_val * vy;

    const int trans_idx = (b * N * T + n * T + t) * D;
    output[trans_idx] = trans_x;
    output[trans_idx + 1] = trans_y;
    output[trans_idx + 2] = z;
    output[trans_idx + 3] = dx;
    output[trans_idx + 4] = dy;
    output[trans_idx + 5] = dz;
    output[trans_idx + 6] = yaw - center_yaw;
    output[trans_idx + 7] = trans_vx;
    output[trans_idx + 8] = trans_vy;
    output[trans_idx + 9] = is_valid;
  }
}

__device__ void generate_onehot_mask(
  const int B, const int N, const int T, const int C, const int sdc_index, const int * target_index,
  const int * object_type_index, float * onehot)
{
  // output [B * N * T * (C + 2)]
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;

  if (b < B && n < N && t < T) {
    int idx = b * N * T + n * T + t;

    // set default value to 0.0
    onehot[idx] = 0.0f;
    onehot[idx * (C + 2) + object_type_index[n]] = 1.0f;

    if (target_index[b] == n) {
      onehot[idx * (C + 2) + C] = 1.0f;
    }

    if (sdc_index == n) {
      onehot[idx * (C + 2) + C + 1] = 1.0f;
    }
  }
}

__device__ void generate_embedding(
  const int B, const int N, const int T, const int D, const float * timestamps,
  const float * trajectory, float * time_embed, float * heading_embed)
{
  // time_embed [B * N * T * (T + 1)]
  // heading_embed [B * N * T * 2]
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;

  if (b < B && n < N && t < T) {
    const int idx = b * N * T + n * T + t;
    time_embed[idx] = 0.0f;
    time_embed[idx * (T + 1) + t] = 1.0f;
    time_embed[idx * (T + 1) + T] = timestamps[t];

    const float yaw = trajectory[idx * D + 6];
    heading_embed[idx * 2] = std::sin(yaw);
    heading_embed[idx * 2 + 1] = std::cos(yaw);
  }
}

__device__ void extract_last_pos(
  const int B, const int N, const int T, const int D, const float * trajectory, float * output)
{
  // output [B * N * 3]
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;

  if (b < B && t == T - 1) {
    const int idx = b * N * T + n * T + t;
    const int out_idx = b * N + n;
    output[out_idx] = 0.0f;
    output[out_idx * 3] = trajectory[idx * D];
    output[out_idx * 3 + 1] = trajectory[idx * D + 1];
    output[out_idx * 3 + 2] = trajectory[idx * D + 2];
  }
}

__device__ void concatenate_agent_data(
  const int B, const int N, const int T, const int D, const int C, const float * trajectory,
  const float * onehot, const float * time_embed, const float * heading_embed, float * out_data,
  bool * out_mask)
{
  // out_data [B * N * T * (D - 2 + (C + 2) + (T + 1)]
  // out_mask [B * N * T]
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;
  if (b < B && n < N && t < T) {
    const int idx = b * N * T + n * T + t;
    const int out_idx = idx * (D - 2) + (C + 2) + T + 1;
    out_data[out_idx] = trajectory[idx * D];
    out_data[out_idx + 1] = trajectory[idx * D + 1];
    out_data[out_idx + 2] = trajectory[idx * D + 2];
    out_data[out_idx + 3] = trajectory[idx * D + 3];
    out_data[out_idx + 4] = trajectory[idx * D + 4];
    out_data[out_idx + 5] = trajectory[idx * D + 5];
    for (int c_idx = 0; c_idx < C + 2; ++c_idx) {
      out_data[out_idx + 6 + c_idx] = onehot[idx * (C + 2) + c_idx];
    }
    for (int t_idx = 0; t_idx < T + 1; ++t_idx) {
      out_data[out_idx + C + 2 + t_idx] = time_embed[idx * (T + 1) + t_idx];
    }
    out_data[out_idx + C + 2 + T + 1] = heading_embed[idx * 2];
    out_data[out_idx + C + 2 + T + 1 + 1] = heading_embed[idx * 2 + 1];
    out_data[out_idx + C + 2 + T + 1 + 2] = trajectory[idx * D + 7];
    out_data[out_idx + C + 2 + T + 1 + 3] = trajectory[idx * D + 8];

    out_mask[out_idx] = static_cast<bool>(trajectory[idx * D + D - 1]);
  }
}

__global__ void agentPreprocessKernel(
  const int B, const int N, const int T, const int D, const int C, const int sdc_index,
  const int * target_index, const int * object_type_index, const float * timestamps,
  const float * in_trajectory, float * out_data, bool * out_mask, float * out_last_pos)
{
  extern __shared__ float dst_trajectory[];
  extern __shared__ float onehot[];

  transform_trajectory(B, N, T, D, target_index, in_trajectory, dst_trajectory);

  generate_onehot_mask(B, N, T, C, sdc_index, target_index, object_type_index, onehot);
  __syncthreads();

  extern __shared__ float time_embed[], heading_embed[];
  generate_embedding(B, N, T, D, timestamps, dst_trajectory, time_embed, heading_embed);
  extract_last_pos(B, N, T, D, dst_trajectory, out_last_pos);
  __syncthreads();

  concatenate_agent_data(
    B, N, T, D, C, dst_trajectory, onehot, time_embed, heading_embed, out_data, out_mask);
}