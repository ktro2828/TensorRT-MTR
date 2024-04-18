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

#ifndef PREPROCESS__POLYLINE_PREPROCESS_KERNEL_CUH_
#define PREPROCESS__POLYLINE_PREPROCESS_KERNEL_CUH_

/**
 * @brief Transform to target agent coordinates system.
 *
 * @param K The number of polylines.
 * @param P The number of points contained in each polyline.
 * @param PointDim The number of point state dimensions.
 * @param in_polyline Source polylines, in shape [K*P*PointDim].
 * @param B The number of target agents.
 * @param AgentDim The number of agent state dimensions.
 * @param target_state Source target state at the latest timestamp, in shape [B*AgentDim].
 * @param out_polyline Output polylines, in shape [B*K*P*(PointDim+2)].
 * @param out_polyline_mask Output polyline mask, in shape [B*K*P].
 */
__global__ void transformPolylineKernel(
  const int K, const int P, const int PointDim, const float * in_polyline, const int B,
  const int AgentDim, const float * target_state, float * out_polyline, bool * out_polyline_mask);

/**
 * @brief Set the previous xy position at the end of element.
 *
 * @param B The number of target agents.
 * @param K The number of polylines.
 * @param P The number of points contained in each polyline.
 * @param D The number of point dimensions.
 * @param mask The polyline mask, in shape [B*K*P].
 * @param polyline The container of polylines, in shape [B*K*P*D]
 */
__global__ void setPreviousPositionKernel(
  const int B, const int K, const int P, const int D, const bool * mask, float * polyline);

/**
 * @brief Calculate distances from targets to polylines.
 *
 * @note `inPolyline` must have been transformed to target coordinates system.
 *
 * @param B The number of target agents.
 * @param L The number of polylines.
 * @param P The number of points contained in each polyline.
 * @param AgentDim The number agent state dimensions.
 * @param targetState Source target agent state, in shape [B*AgentDim].
 * @param PointDim The number of point dimensions.
 * @param inPolyline Source polyline, in shape [B*L*P*PointDim].
 * @param inPolylineMask Source polyline mask, in shape [B*K*P].
 * @param outDistance Output distance, in shape [B*L].
 */
__global__ void calculateCenterDistanceKernel(
  const int B, const int L, const int P, const int AgentDim, const float * targetState,
  const int PointDim, const float * inPolyline, const bool * inPolylineMask, float * outDistance);

/**
 * @brief Extract K polylines with the smallest distances.
 *
 * @note This kernel is implemented using shared memory.
 *
 * @param K The number elements to be extracted (K <= L).
 * @param B The number of target agents.
 * @param L The number of polylines.
 * @param P The number of points contained in each polyline.
 * @param D The number of point dimensions.
 * @param inPolyline Source polyline, in shape [B*L*P*D].
 * @param inPolylineMask Source polyline mask, in shape [B*L*P].
 * @param inDistance Source polyline distances, in shape [B*L].
 * @param outPolyline Output polyline, in shape [B*K*P*D].
 * @param outPolylineMask Output polyline mask, in shape [B*K*P].
 */
__global__ void extractTopKPolylineKernel(
  const int K, const int B, const int L, const int P, const int D, const float * inPolyline,
  const bool * inPolylineMask, const float * inDistance, float * outPolyline,
  bool * outPolylineMask);

/**
 * @brief Calculate the magnitudes of polylines.
 *
 * @param B The number of target agents.
 * @param K The number of polylines.
 * @param P The number of points contained in each polyline.
 * @param D The number of point dimensions.
 * @param polyline Source polylines, in shape [B*K*P*(D + 2)].
 * @param mask Source polyline masks, in shape [B*K*P].
 * @param center Output magnitudes of polylines, in shape [B*K*3].
 */
__global__ void calculatePolylineCenterKernel(
  const int B, const int K, const int P, const int PointDim, const float * polyline,
  const bool * mask, float * center);

/**
 * @brief In cases of the number of batch polylines (L) is greater than K,
 *  extracts the topK elements.
 *
 * @param L The number of source polylines.
 * @param K The number of polylines expected as the model input.
 * @param P The number of points contained in each polyline.
 * @param PointDim The number of point state dimensions.
 * @param AgentDim The number of agent state dimensions.
 * @param in_polyline Source polylines, in shape [L*P*PointDim].
 * @param B The number of target agents.
 * @param target_state Target agent state at the latest timestamp, in shape [B, AgentDim].
 * @param out_polyline Output polylines, in shape [B*K*P*(PointDim+2)].
 * @param out_polyline_mask Output polyline masks, in shape [B*K*P].
 * @param out_polyline_center Output magnitudes of each polyline with respect to target coords,
 *  in shape [B*K*3].
 * @param stream CUDA stream.
 * @return cudaError_t
 */
cudaError_t polylinePreprocessWithTopkLauncher(
  const int L, const int K, const int P, const int PointDim, const float * in_polyline, const int B,
  const int AgentDim, const float * target_state, float * out_polyline, bool * out_polyline_mask,
  float * out_polyline_center, cudaStream_t stream);

/**
 * @brief Do preprocess for polyline if the number of batched polylines is K.
 *
 * @param K The number of polylines.
 * @param P The number of points contained in each polyline.
 * @param PointDim The number of point state dimensions.
 * @param in_polyline Source polylines, in shape [K*P*PointDim].
 * @param B The number of target agents.
 * @param AgentDim The number of agent state dimensions.
 * @param target_state Target agent state at the latest timestamp, in shape [B, AgentDim].
 * @param out_polyline Output polylines, in shape [B*K*P*(PointDim + 2)].
 * @param out_polyline_mask Output polyline masks, in shape [B*K*P].
 * @param out_polyline_center Output magnitudes of each polyline with respect to target coords,
 *  in shape [B*K*3].
 * @param stream CUDA stream.
 * @return cudaError_t
 */
cudaError_t polylinePreprocessLauncher(
  const int K, const int P, const int PointDim, const float * in_polyline, const int B,
  const int AgentDim, const float * target_state, float * out_polyline, bool * out_polyline_mask,
  float * out_polyline_center, cudaStream_t stream);

#endif  // PREPROCESS__POLYLINE_PREPROCESS_KERNEL_CUH_
