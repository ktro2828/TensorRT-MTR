#ifndef KNN__TRT_KNN_BATCH_MLOGK_KERNEL_HPP_
#define KNN__TRT_KNN_BATCH_MLOGK_KERNEL_HPP_

#include <cuda_runtime.h>

cudaError_t KnnBatchMlogKLauncher(
  const int32_t n, const int32_t m, const int32_t k, const float * xyz, const float * query_xyz,
  const int * batch_idx, const int * query_batch_offsets, int * output, cudaStream_t stream);

#endif  // KNN__TRT_KNN_BATCH_MLOGK_KERNEL_HPP_