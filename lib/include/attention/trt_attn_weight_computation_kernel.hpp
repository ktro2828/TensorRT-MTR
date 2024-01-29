#ifndef TRT_ATTN_WEIGHT_COMPUTATION_KERNEL_HPP
#define TRT_ATTN_WEIGHT_COMPUTATION_KERNEL_HPP

#include <cuda_runtime.h>

/**
 * @brief The launcher to invoke attention weight computation kernel.
 *
 * @tparam T
 * @param b
 * @param total_query_num
 * @param local_size
 * @param total_key_num
 * @param nhead
 * @param hdim
 * @param query_batch_cnt
 * @param key_batch_cnt
 * @param index_pair_batch
 * @param index_pair
 * @param query_features
 * @param key_features
 * @param output
 */
template <typename T>
void AttentionWeightComputationLauncher(
  int b, int total_query_num, int local_size, int total_key_num, int nhead, int hdim,
  const int * query_batch_cnt, const int * key_batch_cnt, const int * index_pair_batch,
  const int * index_pair, const T * query_features, const T * key_features, T * output,
  cudaStream_t stream);

#endif  // TRT_ATTN_WEIGHT_COMPUTATION_KERNEL_HPP