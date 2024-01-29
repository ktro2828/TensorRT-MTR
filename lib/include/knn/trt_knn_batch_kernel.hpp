#ifndef TRT_KNN_BATCH_KERNEL_HPP
#define TRT_KNN_BATCH_KERNEL_HPP

#include <cuda_runtime.h>

template <typename T>
cudaError_t KnnBatchLauncher(
  const int n, const int m, const int k, const T * xyz, const T * query_xyz, const int * batch_idx,
  const int * query_batch_offsets, int * output, cudaStream_t stream);

#endif  // TRT_KNN_BATCH_KERNEL_HPP