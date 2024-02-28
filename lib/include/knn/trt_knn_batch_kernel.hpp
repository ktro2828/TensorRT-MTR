#ifndef KNN__TRT_KNN_BATCH_KERNEL_HPP_
#define KNN__TRT_KNN_BATCH_KERNEL_HPP_

#include <cuda_runtime.h>

template <typename T>
cudaError_t KnnBatchLauncher(
  const int32_t n, const int32_t m, const int32_t k, const T * xyz, const T * query_xyz,
  const int * batch_idx, const int * query_batch_offsets, int * output, cudaStream_t stream);

#endif  // KNN__TRT_KNN_BATCH_KERNEL_HPP_