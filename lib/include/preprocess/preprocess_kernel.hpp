#ifndef PREPROCESS__PREPROCESS_KERNEL_HPP_
#define PREPROCESS__PREPROCESS_KERNEL_HPP_

#include <cuda_runtime.h>

/**
 * @brief
 *
 * @param B
 * @param N
 * @param T
 * @param D
 * @param sdc_index
 * @param target_index
 * @param object_type_index
 * @param timestamps
 * @param in_trajectory
 * @param out_data
 * @param out_mask
 * @param out_last_pos
 * @param stream
 * @return cudaError_t
 */
cudaError_t agentPreprocessLauncher(
  const int B, const int N, const int T, const int D, const int C, const int sdc_index,
  const int * target_index, const int * object_type_index, const float * timestamps,
  const float * in_trajectory, float * out_data, bool * out_mask, float * out_last_pos,
  cudaStream_t stream);

/**
 * @brief
 *
 * @param polylines
 * @param out_polylines
 * @param out_mask
 * @return cudaError_t
 */
cudaError_t polylinePreprocessLauncher(
  const float * polylines, float * out_polylines, bool * out_mask);

#endif  // PREPROCESS__PREPROCESS_KERNEL_HPP_