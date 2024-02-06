#ifndef PREPROCESS__PREPROCESS_KERNEL_HPP_
#define PREPROCESS__PREPROCESS_KERNEL_HPP_

#include <cuda_runtime.h>

/**
 * @brief
 *
 * @param num_targets
 * @param num_agents
 * @param sdc_index
 * @param target_index
 * @param agent_trajectory
 * @param out_trajectory
 * @param out_mask
 * @return cudaError_t
 */
cudaError_t generateTargetCentricTrajectoryLauncher(
  const int num_targets, const int num_agents, const int sdc_index, const int * target_index,
  const float * agent_trajectory, float * out_trajectory, bool * out_mask);

/**
 * @brief
 *
 * @param polylines
 * @param out_polylines
 * @param out_mask
 * @return cudaError_t
 */
cudaError_t generateTargetCentricPolylineLauncher(
  const float * polylines, float * out_polylines, bool * out_mask);

#endif  // PREPROCESS__PREPROCESS_KERNEL_HPP_