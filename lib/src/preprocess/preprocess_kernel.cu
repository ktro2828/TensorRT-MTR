#include "preprocess/preprocess_kernel.hpp"

#include <cmath>

#define THREADS_PER_BLOCK 256;

/**
 * @brief Rotate points along z axis.
 *
 * @param B Batch size.
 * @param N The number of points.
 * @param C The number of the other attribtues.
 * @param src Source points, in shape (B * N * 3 + C).
 * @param angles Source angles [deg], in shape (B).
 * @param dst Destination points, in shape (B * N * 3 + C).
 */
__global__ void rotate_points_kernel(
  const int B, const int N, const int C, const float * src, const float * angles, float * dst)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B * N) {
    const int b = idx / N;

    const float angle = angles[b];
    const float cos_val = std::cos(angle);
    const float sin_val = std::sin(angle);

    const float x = src[idx * 3];
    const float y = src[idx * 3 + 1];

    dst[idx * (3 + C)] = cos_val * x - sin_val * y;
    dst[idx * (3 + C) + 1] = sin_val * x + cos_val * y;
    dst[idx * (3 + C) + 2] = src[idx * 3 + 2];
  }
}

/**
 * @brief Transform input trajectory from world to each target centric coords.
 *
 * @param B The number of targets (=B).
 * @param N The number of agents(=N).
 * @param T The number of timestamps(=T).
 * @param D The number of attributes(=D).
 * @param in_trajectory Source trajectories, in shape (N * T * D).
 *  ordering (x, y, z, dx, dy, dz, yaw, vx, vy, valid).
 * @param center_xyz Source center xyz points, in shape (B * 3).
 * @param center_yaw Source headings, in shape (B).
 * @param output Output trajectories, in shape (B * N * T * D).
 */
__global__ void transform_trajectory_to_target_centric_kernel(
  const int B, const int N, const int T, const int D, const float * in_trajectory,
  const float * center_xyz, const float * center_yaw, float * output)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * T) {
    const float x = in_trajectory[idx * D];
    const float y = in_trajectory[idx * D + 1];
    const float z = in_trajectory[idx * D + 2];
    const float vx = in_trajectory[idx * D + 7];
    const float vy = in_trajectory[idx * D + 8];

    // transform for each target
    for (int b = 0; b < B; ++b) {
      const float cos_val = std::cos(center_yaw[b]);
      const float sin_val = std::sin(center_yaw[b]);

      // transform
      const float trans_x = cos_val * x - sin_val * y + center_xyz[b * 3];
      const float trans_y = sin_val * x + cos_val * y + center_xyz[b * 3 + 1];
      const float trans_vx = cos_val * vx - sin_val * vy + center_xyz[b * 3];
      const float trans_vy = sin_val * vx + cos_val * vy + center_xyz[b * 3 + 1];

      output[(b * N * T + idx) * D] = trans_x;
      output[(b * N * T + idx) * D + 1] = trans_y;
      output[(b * N * T + idx) * D + 2] = z;
      output[(b * N * T + idx) * D + 7] = trans_vx;
      output[(b * N * T + idx) * D + 8] = trans_vy;
    }
  }
}

/**
 * @brief Generate onehot mask.
 *
 * @param B The number of targets.
 * @param N The number of agents.
 * @param T The number of timestamps.
 * @param sdc_index The index of ego vehicle.
 * @param object_types The list of objec types, in shape (N).
 *   (0: vehicle, 1: pedestrian, 2:cyclist).
 * @param center_index The list of target object indices, in shape (B).
 * @param output Output onehot mask, in shape (B * N * T * 5).
 *
 * @example
 * ```
 * dim3 threadsPerBlock(8, 8, 8)
 * dim3 numBlocks((B + threadsPerBlock.x - 1) / threadsPerBlock.x,
 *                (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
 *                (T + threadsPerBlock.z - 1) / threadsPerBlock.z)
 * generate_onehot_mask<<<numBlocks, threadsPerBlock>>>
 * ```
 */
__global__ void generate_onehot_mask_kernel(
  const int B, const int N, const int T, const int sdc_index, const int * object_types,
  const int * center_index, float * output)
{
  // output [B * N * T * 5]
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;

  if (b < B && n < N && t < T) {
    int idx = b * N * T + n * T + t;

    // set default value to 0.0
    output[idx] = 0.f;

    if (object_types[n] == 0 && t == 0) {
      output[idx] = 1.f;
    } else if (object_types[n] == 1 && t == 1) {
      output[idx] = 1.f;
    } else if (object_types[n] == 2 && t == 1) {
      output[idx] = 1.f;
    }

    if (center_index[b] == n && t == 3) {
      output[idx] = 1.f;
    }

    if (sdc_index == n && t == 4) {
      output[idx] = 1.f;
    }
  }
}

/**
 * @brief TODO
 *
 * @param B
 * @param N
 * @param T
 * @param timestamps
 * @param trajectory
 * @param out_time
 * @param out_yaw
 * @param out_accel
 * @return __global__
 */
__global__ void generate_agent_embedding_kernel(
  const int B, const int N, const int T, const float * timestamps, const float * trajectory,
  float * out_time, float * out_yaw, float * out_accel)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;

  if (b < B && n < N && t < T) {
    const int time_idx = b * N * T + n * T + t;
    // === time embeding ===
    // set default value to 0.0
    out_time[time_idx] = 0.f;
    out_time[time_idx] = 1.0f;
    out_time[time_idx + T] = timestamps[t];

    // === yaw embeding ===
    // set default value to 0.0
    const int yaw_idx = b * N * T * 2 + n * T * 2 + t * 2;
    const float yaw = trajectory[b * N * T * 7 + n * T * 7 + t * 7 + 6];
    out_yaw[yaw_idx] = std::sin(yaw);
    out_yaw[yaw_idx + 1] = std::cos(yaw);

    // === accel ===
    const int acc_idx = b * N * T * 2 + n * T * 2 + t * 2;
    out_accel[acc_idx] = (trajectory[acc_idx] - ((t > 0) ? trajectory[acc_idx - 2] : 0.f)) / 0.1f;
    out_accel[acc_idx + 1] =
      (trajectory[acc_idx + 1] - ((t > 0) ? trajectory[acc_idx - 1] : 0.f)) / 0.1f;
    if (t == 0) {
      out_accel[acc_idx] = out_accel[acc_idx + 2];
      out_accel[acc_idx + 1] = out_accel[acc_idx + 3];
    }
  }
}

cudaError_t generateTargetCentricTrajectoryLauncher(
  const int num_targets, const int num_agents, const int sdc_index, const int * target_index,
  const float * agent_trajectory, float * out_trajectory, bool * out_mask)
{
  return cudaGetLastError();
}

cudaError_t generateTargetCentricPolylineLauncher(
  const float * polylines, float * out_polylines, bool * out_mask)
{
  return cudaGetLastError();
}
