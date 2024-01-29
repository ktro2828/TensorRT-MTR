#ifndef TRT_ATTN_VALUE_COMPUTATION_KERNEL_HPP
#define TRT_ATTN_VALUE_COMPUTATION_KERNEL_HPP

#include <cuda_runtime.h>

/**
 * @brief The launcher to invoke attention value computation kernel.
 *
 * @tparam T
 * @param b
 * @param total_query_num
 * @param local_size
 * @param total_value_num
 * @param nhead
 * @param hdim
 * @param query_batch_cnt
 * @param key_batch_cnt
 * @param index_pair_batch
 * @param index_pair
 * @param attn_weight
 * @param value_features
 * @param output
 */
template <typename T>
void AttentionValueComputationLauncher(
  int b, int total_query_num, int local_size, int total_value_num, int nhead, int hdim,
  const int * query_batch_cnt, const int * key_batch_cnt, const int * index_pair_batch,
  const int * index_pair, const T * attn_weight, const T * value_features, T * output,
  cudaStream_t stream);

#endif  // TRT_ATTN_VALUE_COMPUTATION_KERNEL_HPP