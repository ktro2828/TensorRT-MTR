// Copyright 2024 Kotaro Uetake
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mtr/cuda_helper.hpp"
#include "preprocess/polyline_preprocess_kernel.cuh"

#include <float.h>

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

  bool is_valid = false;
  for (size_t i = 0; i < PointDim - 1; ++i) {
    if (out_polyline[out_idx + i] != 0.0f) {
      is_valid = true;
    }
  }
  int out_mask_idx = b * K * P + k * P + p;
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

__global__ void calculateCenterDistanceKernel(
  const int B, const int L, const int P, const int AgentDim, const float * targetState,
  const int PointDim, const float * inPolyline, const bool * inPolylineMask, float * outDistance)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int l = blockIdx.y * blockDim.y + threadIdx.y;
  if (b >= B || l >= L) {
    return;
  }

  // calculate polyline center
  float sumX = 0.0f, sumY = 0.0f;
  int numValid = 0;
  for (int p = 0; p < P; ++p) {
    int idx = b * L * P + l * P + p;
    float x = inPolyline[idx * PointDim];
    float y = inPolyline[idx * PointDim + 1];
    if (inPolylineMask[idx]) {
      sumX += x;
      sumY += y;
      ++numValid;
    }
  }
  float centerX = sumX / fmaxf(1.0f, numValid);
  float centerY = sumY / fmaxf(1.0f, numValid);

  outDistance[b * L + l] = sqrtf(powf(centerX, 2) + powf(centerY, 2));
}

__global__ void extractTopKPolylineKernel(
  const int K, const int B, const int L, const int P, const int D, const float * inPolyline,
  const bool * inPolylineMask, const float * inDistance, float * outPolyline,
  bool * outPolylineMask)
{
  int b = blockIdx.x;  // Batch index
  extern __shared__ float distances[];

  // Load distances into shared memory
  int tid = threadIdx.x;
  if (tid < L) {
    distances[tid] = inDistance[b * L + tid];
  }
  __syncthreads();

  // Simple selection of the smallest K distances
  // (this part should be replaced with a more efficient sorting/selecting algorithm)
  for (int k = 0; k < K; k++) {
    float minDistance = FLT_MAX;
    int minIndex = -1;

    for (int l = 0; l < L; l++) {
      if (distances[l] < minDistance) {
        minDistance = distances[l];
        minIndex = l;
      }
    }

    if (tid == k) {  // this thread will handle copying the k-th smallest polyline
      for (int p = 0; p < P; p++) {
        for (int d = 0; d < D; d++) {
          outPolyline[b * K * P * D + k * P * D + p * D + d] =
            inPolyline[b * L * P * D + minIndex * P * D + p * D + d];
        }
        outPolylineMask[b * K * P + k * P + p] = inPolylineMask[b * L * P + minIndex * P + p];
      }
    }
    distances[minIndex] = FLT_MAX;  // exclude this index from future consideration
  }
}

__global__ void calculatePolylineCenterKernel(
  const int B, const int K, const int P, const int PointDim, const float * polyline,
  const bool * mask, float * center)
{
  // --- pseudo code ---
  // sum = (polylines[:, :, :, 0:3] * mask[:, :, :, None]).sum(dim=2)
  // center = sum / clampMIN(mask.sum(dim=2), min=1.0)
  // -------------------

  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (b >= B || k >= K) {
    return;
  }

  // initialize with 0.0
  int center_idx = (b * K + k) * 3;
  for (int d = 0; d < 3; ++d) {
    center[center_idx + d] = 0.0f;
  }

  float sum_xyz[3] = {0.0f, 0.0f, 0.0f};
  int count = 0;
  for (int p = 0; p < P; ++p) {
    int src_idx = b * K * P + k * P + p;
    if (mask[src_idx]) {
      for (int d = 0; d < 3; ++d) {
        sum_xyz[d] += polyline[src_idx * PointDim + d];
      }
      ++count;
    }
  }
  count = max(count, 1);

  for (int d = 0; d < 3; ++d) {
    center[center_idx + d] = sum_xyz[d] / static_cast<float>(count);
  }
}

cudaError_t polylinePreprocessWithTopkLauncher(
  const int L, const int K, const int P, const int PointDim, const float * in_polyline, const int B,
  const int AgentDim, const float * target_state, float * out_polyline, bool * out_polyline_mask,
  float * out_polyline_center, cudaStream_t stream)
{
  if (L < K) {
    std::cerr << "L must be greater than K, but got L: " << L << ", K: " << K << std::endl;
    return cudaError_t::cudaErrorInvalidValue;
  }

  float *tmpPolyline, *tmpDistance;
  bool * tmpPolylineMask;
  CHECK_CUDA_ERROR(cudaMallocAsync(&tmpPolyline, sizeof(float) * B * L * P * PointDim, stream));
  CHECK_CUDA_ERROR(cudaMallocAsync(&tmpPolylineMask, sizeof(bool) * B * L * P, stream));
  CHECK_CUDA_ERROR(cudaMallocAsync(&tmpDistance, sizeof(float) * B * L, stream));

  // TODO: update the number of blocks and threads to guard from `cudaErrorIllegalAccess`
  constexpr int threadsPerBlock = 256;
  const dim3 block3dl(B, L / threadsPerBlock, P);
  transformPolylineKernel<<<block3dl, threadsPerBlock, 0, stream>>>(
    L, P, PointDim, in_polyline, B, AgentDim, target_state, tmpPolyline, tmpPolylineMask);

  const dim3 block2dl(B, L / threadsPerBlock);
  calculateCenterDistanceKernel<<<block2dl, threadsPerBlock, 0, stream>>>(
    B, L, P, AgentDim, target_state, PointDim, tmpPolyline, tmpPolylineMask, tmpDistance);

  const size_t sharedMemSize = sizeof(float) * K;
  extractTopKPolylineKernel<<<B, threadsPerBlock, sharedMemSize, stream>>>(
    K, B, L, P, PointDim, tmpPolyline, tmpPolylineMask, tmpDistance, out_polyline,
    out_polyline_mask);

  const dim3 block3dk(B, K / threadsPerBlock, P);
  setPreviousPositionKernel<<<block3dk, threadsPerBlock, 0, stream>>>(
    B, K, P, PointDim, out_polyline_mask, out_polyline);

  const dim3 block2dk(B, K / threadsPerBlock);
  calculatePolylineCenterKernel<<<block2dk, threadsPerBlock, 0, stream>>>(
    B, K, P, PointDim, out_polyline, out_polyline_mask, out_polyline_center);

  CHECK_CUDA_ERROR(cudaFree(tmpPolyline));
  CHECK_CUDA_ERROR(cudaFree(tmpPolylineMask));
  CHECK_CUDA_ERROR(cudaFree(tmpDistance));

  return cudaGetLastError();
}

cudaError_t polylinePreprocessLauncher(
  const int K, const int P, const int PointDim, const float * in_polyline, const int B,
  const int AgentDim, const float * target_state, float * out_polyline, bool * out_polyline_mask,
  float * out_polyline_center, cudaStream_t stream)
{
  // TODO: update the number of blocks and threads to guard from `cudaErrorIllegalAccess`
  constexpr int threadsPerBlock = 256;
  const dim3 block3d(B, (K - 1) / threadsPerBlock, P);

  transformPolylineKernel<<<block3d, threadsPerBlock, 0, stream>>>(
    K, P, PointDim, in_polyline, B, AgentDim, target_state, out_polyline, out_polyline_mask);

  setPreviousPositionKernel<<<block3d, threadsPerBlock, 0, stream>>>(
    B, K, P, PointDim, out_polyline_mask, out_polyline);

  const dim3 block2d(B, (K - 1) / threadsPerBlock);
  calculatePolylineCenterKernel<<<block2d, threadsPerBlock, 0, stream>>>(
    B, K, P, PointDim, out_polyline, out_polyline_mask, out_polyline_center);

  return cudaGetLastError();
}
