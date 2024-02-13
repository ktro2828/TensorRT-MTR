#ifndef PREPROCESS__PREPROCESS_KERNEL_HPP_
#define PREPROCESS__PREPROCESS_KERNEL_HPP_

#include <cuda_runtime.h>

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