#ifndef PREPROCESS__AGENT_PREPROCESS_KERNEL_HPP_
#define PREPROCESS__AGENT_PREPROCESS_KERNEL_HPP_

#include <cuda_runtime.h>

/**
 * @brief Preprocess kernel for agent data.
 *
 * @param B The number of target agents.
 * @param N The number of all agents.
 * @param T The number of timestamps.
 * @param D The number of agent state dimensions.
 * @param C The number of agent classes.
 * @param sdc_index The index of ego.
 * @param target_index The array of target indices, in shape [B].
 * @param object_type_index The array of agent class indices, in shape [N].
 *  (e.g. 0: VEHICLE, 1:PEDESTRIAN, 2: CYCLIST).
 * @param timestamps The array of timestamps, in shape [T].
 * @param in_trajectory The array of trajectory, in shape [N*T*D].
 * @param out_data Output trajectory data, in shape [B*N*T*(D+C+T+5)].
 * @param out_mask Output mask trajectory data, in shape [B*N*T].
 * @param out_last_pos Output last position, in shape [B*N*3].
 */
__global__ void agentPreprocessKernel(
  const int B, const int N, const int T, const int D, const int C, const int sdc_index,
  const int * target_index, const int * object_type_index, const float * timestamps,
  const float * in_trajectory, float * out_data, bool * out_mask, float * out_last_pos);

/**
 * @brief Preprocess kernel for agent data.
 *
 * @param B The number of target agents.
 * @param N The number of all agents.
 * @param T The number of timestamps.
 * @param D The number of agent state dimensions.
 * @param C The number of agent classes.
 * @param sdc_index The index of ego.
 * @param target_index The array of target indices, in shape [B].
 * @param object_type_index The array of agent class indices, in shape [N].
 *  (e.g. 0: VEHICLE, 1:PEDESTRIAN, 2: CYCLIST).
 * @param timestamps The array of timestamps, in shape [T].
 * @param in_trajectory The array of trajectory, in shape [N*T*D].
 * @param out_data Output trajectory data, in shape [B*N*T*(D+C+T+5)].
 * @param out_mask Output mask trajectory data, in shape [B*N*T].
 * @param out_last_pos Output last position, in shape [B*N*3].
 * @param stream CUDA stream.
 * @return cudaError_t CUDA error code.
 */
cudaError_t agentPreprocessLauncher(
  const int B, const int N, const int T, const int D, const int C, const int sdc_index,
  const int * target_index, const int * object_type_index, const float * timestamps,
  const float * in_trajectory, float * out_data, bool * out_mask, float * out_last_pos,
  cudaStream_t stream);

#endif  // PREPROCESS__AGENT_PREPROCESS_KERNEL_HPP_