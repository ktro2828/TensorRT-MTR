#include "attention/trt_attn_weight_computation_kernel.hpp"

#include <stdio.h>

template <typename T, unsigned int d>
__global__ void attentionWeightComputationKernel(
  const int32_t B, const int32_t Q, const int32_t L, const int32_t K, const int32_t numHead,
  const int32_t headDim, const int * queryBatchCnt, const int * keyBatchCnt,
  const int * indexPairBatch, const int * indexPair, const T * queryFeature, const T * keyFeature,
  T * output)
{
  const int query_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int local_key_idx = threadIdx.x;

  const int index = query_idx * L + local_key_idx;

  if (query_idx >= Q || head_idx >= numHead || local_key_idx >= L) {
    return;
  }

  // build shared query features
  __shared__ T sharedQueryFeature[d];
  for (int i = local_key_idx; i < headDim; i += blockDim.x) {
    sharedQueryFeature[i] = queryFeature[query_idx * numHead * headDim + head_idx * headDim + i];
  }
  __syncthreads();

  if (indexPair[index] < 0) {
    // ignore index
    return;
  }

  // get real key index
  const int batch_idx = indexPairBatch[query_idx];
  int key_start_idx = 0;
  for (int i = 0; i < batch_idx; ++i) {
    key_start_idx += keyBatchCnt[i];
  }
  key_start_idx += indexPair[index];

  // get key features
  keyFeature += key_start_idx * numHead * headDim + head_idx * headDim;
  output += index * numHead + head_idx;

  T attn_weight = 0;
  for (int i = 0; i < headDim; ++i) {
    // TODO: fix bug
    // attn_weight += keyFeature[i] * sharedQueryFeature[i];
  }
  output[0] = attn_weight;
}

template <typename T>
cudaError_t AttentionWeightComputationLauncher(
  const int32_t B, const int32_t Q, const int32_t L, const int32_t K, const int32_t numHead,
  const int32_t headDim, const int * queryBatchCnt, const int * keyBatchCnt,
  const int * indexPairBatch, const int * indexPair, const T * queryFeature, const T * keyFeature,
  T * output, cudaStream_t stream)
{
  if (headDim > 150) {
    return cudaError::cudaErrorInvalidValue;
  }

  dim3 blocks(Q, numHead);
  dim3 threads(L);

  switch (headDim) {
    case 16:
      attentionWeightComputationKernel<T, 16><<<blocks, threads, 0, stream>>>(
        B, Q, L, K, numHead, headDim, queryBatchCnt, keyBatchCnt, indexPairBatch, indexPair,
        queryFeature, keyFeature, output);
      break;
    case 24:
      attentionWeightComputationKernel<T, 24><<<blocks, threads, 0, stream>>>(
        B, Q, L, K, numHead, headDim, queryBatchCnt, keyBatchCnt, indexPairBatch, indexPair,
        queryFeature, keyFeature, output);
      break;
    case 32:
      attentionWeightComputationKernel<T, 32><<<blocks, threads, 0, stream>>>(
        B, Q, L, K, numHead, headDim, queryBatchCnt, keyBatchCnt, indexPairBatch, indexPair,
        queryFeature, keyFeature, output);
      break;
    case 48:
      attentionWeightComputationKernel<T, 48><<<blocks, threads, 0, stream>>>(
        B, Q, L, K, numHead, headDim, queryBatchCnt, keyBatchCnt, indexPairBatch, indexPair,
        queryFeature, keyFeature, output);
      break;
    case 64:
      attentionWeightComputationKernel<T, 64><<<blocks, threads, 0, stream>>>(
        B, Q, L, K, numHead, headDim, queryBatchCnt, keyBatchCnt, indexPairBatch, indexPair,
        queryFeature, keyFeature, output);
      break;
    case 128:
      attentionWeightComputationKernel<T, 128><<<blocks, threads, 0, stream>>>(
        B, Q, L, K, numHead, headDim, queryBatchCnt, keyBatchCnt, indexPairBatch, indexPair,
        queryFeature, keyFeature, output);
      break;
    default:
      attentionWeightComputationKernel<T, 150><<<blocks, threads, 0, stream>>>(
        B, Q, L, K, numHead, headDim, queryBatchCnt, keyBatchCnt, indexPairBatch, indexPair,
        queryFeature, keyFeature, output);
      break;
  }
  return cudaGetLastError();
}

template cudaError_t AttentionWeightComputationLauncher<float>(
  const int32_t B, const int32_t Q, const int32_t L, const int32_t K, const int32_t numHead,
  const int32_t headDim, const int * queryBatchCnt, const int * keyBatchCnt,
  const int * indexPairBatch, const int * indexPair, const float * queryFeature,
  const float * keyFeature, float * output, cudaStream_t stream);
