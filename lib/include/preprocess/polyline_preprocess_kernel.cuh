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
 * @param polyline_mask The polyline mask, in shape [B*K*P].
 * @param out_polyline Output polylines, in shape [B*K*P*(PointDim+2)].
 *  in shape [B*K*3].
 */
__global__ void transformPolylineKernel(
  const int K, const int P, const int PointDim, const float * in_polyline, const int B,
  const int AgentDim, const float * target_state, const bool * polyline_mask, float * out_polyline);

/**
 * @brief Set the previous xy position at the end of element.
 *
 * @param B The number of target agents.
 * @param K The number of polylines.
 * @param P The number of points contained in each polyline.
 * @param D The number of point dimensions.
 * @param mask The polyline mask, in shape [B*K*P].
 * @param polyline The container of polylines, in shape [B*K*P*D]
 * @return __global__
 */
__global__ void setPreviousPositionKernel(
  const int B, const int K, const int P, const int D, const bool * mask, float * polyline);

/**
 * @brief In cases of the number of batch polylines (L) is greater than K,
 *  extacts the topK elements.
 *
 * @param L The number of source polylines.
 * @param K The number of polylines expected as the model input.
 * @param P The number of points contained in each polyline.
 * @param PointDim The number of point state dimensions.
 * @param AgentDim The number of agent state dimensions.
 * @param in_polyline Source polylines, in shape [L*P*PointDim].
 * @param B The number of target agents.
 * @param target_state Target agent state at the latest timestamp, in shape [B, AgentDim].
 * @param topk_index A container to store topK indices, in shape [K].
 * @param out_polyline Output polylines, in shape [B*K*P*(PointDim+2)].
 * @param out_polyline_mask Output polyline masks, in shape [B*K*P].
 * @param out_polyline_center Output magnitudes of each polyline with respect to target coords,
 *  in shape [B*K*3].
 * @param stream CUDA stream.
 * @return cudaError_t
 */
cudaError_t polylinePreprocessWithTopkLauncher(
  const int L, const int K, const int P, const int PointDim, const float * in_polyline, const int B,
  const int AgentDim, const float * target_state, int * topk_index, float * out_polyline,
  bool * out_polyline_mask, float * out_polyline_center, cudaStream_t stream);

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
