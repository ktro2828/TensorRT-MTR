#ifndef ATTENTION__TRT_ATTN_VALUE_COMPUTATION_KERNEL_HPP_
#define ATTENTION__TRT_ATTN_VALUE_COMPUTATION_KERNEL_HPP_

#include <cuda_runtime.h>

/**
 * @brief The launcher to invoke attention value computation kernel.
 *
 * @tparam T Type of `attnWeight` and `valueFeature`.
 * @param B The size of batch.
 * @param Q The size of query.
 * @param L The size of local.
 * @param K The size of key.
 * @param numHead The number of heads.
 * @param headDim The number of head dimensions.
 * @param queryBatchCnt The number of queries for each batch, in shape [B].
 * @param keyBatchCnt The number of keys for each batch, in shape [B].
 * @param indexPairBatch The indices of batch for corresponding query, in shape [Q].
 * @param indexPair The indices of key for corresponding query, in shape [Q*L].
 * @param attnWeight Source attention weights, in shape [Q*L*numHead].
 * @param valueFeature Source value features, in shape [K*numHead*headDim].
 * @param output Output container, in shape [Q*numHead*headDim].
 * @param stream CUDA stream.
 *
 * @return cudaError_t CUDA error type.
 */
template <typename T>
cudaError_t AttentionValueComputationLauncher(
  const int32_t B, const int32_t Q, const int32_t L, const int32_t K, const int32_t numHead,
  const int32_t headDim, const int * queryBatchCnt, const int * keyBatchCnt,
  const int * indexPairBatch, const int * indexPair, const T * attnWeight, const T * valueFeature,
  T * output, cudaStream_t stream);

#endif  // ATTENTION__TRT_ATTN_VALUE_COMPUTATION_KERNEL_HPP_