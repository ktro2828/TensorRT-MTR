#include "attention/trt_attn_weight_computation_kernel.hpp"

#include <stdio.h>

template <typename T, unsigned int d>
__global__ void attention_weight_computation_kernel(
  const int32_t b, const int32_t total_query_num, const int32_t local_size,
  const int32_t total_key_num, const int32_t nhead, const int32_t hdim, const int * query_batch_cnt,
  const int * key_batch_cnt, const int * index_pair_batch, const int * index_pair,
  const T * query_features, const T * key_features, T * output)
{
  const int query_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int local_key_idx = threadIdx.x;

  const int index = query_idx * local_size + local_key_idx;

  if (query_idx >= total_query_num || head_idx >= nhead || local_key_idx >= local_size) {
    // TODO: WARNING
    return;
  }

  // build shared query features
  __shared__ T shared_query_features[d];
  for (int i = local_key_idx; i < hdim; i += blockDim.x) {
    shared_query_features[i] = query_features[query_idx * nhead * hdim + head_idx * hdim + i];
  }
  __syncthreads();

  if (index_pair[index] < 0) {
    // ignore index
    return;
  }

  // get real key index
  const int batch_idx = index_pair_batch[query_idx];
  int key_start_idx = 0;
  for (int i = 0; i < batch_idx; ++i) {
    key_start_idx += key_batch_cnt[i];
  }
  key_start_idx += index_pair[index];

  // get key features
  key_features += key_start_idx * nhead * hdim + head_idx * hdim;
  output += index * nhead + head_idx;

  T attn_weight = 0;
  for (int i = 0; i < hdim; ++i) {
    // TODO: fix bug
    // attn_weight += key_features[i] * shared_query_features[i];
  }
  output[0] = attn_weight;
}

template <typename T>
cudaError_t AttentionWeightComputationLauncher(
  const int32_t b, const int32_t total_query_num, const int32_t local_size,
  const int32_t total_key_num, const int32_t nhead, const int32_t hdim, const int * query_batch_cnt,
  const int * key_batch_cnt, const int * index_pair_batch, const int * index_pair,
  const T * query_features, const T * key_features, T * output, cudaStream_t stream)
{
  if (hdim > 150) {
    return cudaError::cudaErrorInvalidValue;
  }

  dim3 blocks(total_query_num, nhead);
  dim3 threads(local_size);

  switch (hdim) {
    case 16:
      attention_weight_computation_kernel<T, 16><<<blocks, threads, 0, stream>>>(
        b, total_query_num, local_size, total_key_num, nhead, hdim, query_batch_cnt, key_batch_cnt,
        index_pair_batch, index_pair, query_features, key_features, output);
      break;
    case 24:
      attention_weight_computation_kernel<T, 24><<<blocks, threads, 0, stream>>>(
        b, total_query_num, local_size, total_key_num, nhead, hdim, query_batch_cnt, key_batch_cnt,
        index_pair_batch, index_pair, query_features, key_features, output);
      break;
    case 32:
      attention_weight_computation_kernel<T, 32><<<blocks, threads, 0, stream>>>(
        b, total_query_num, local_size, total_key_num, nhead, hdim, query_batch_cnt, key_batch_cnt,
        index_pair_batch, index_pair, query_features, key_features, output);
      break;
    case 48:
      attention_weight_computation_kernel<T, 48><<<blocks, threads, 0, stream>>>(
        b, total_query_num, local_size, total_key_num, nhead, hdim, query_batch_cnt, key_batch_cnt,
        index_pair_batch, index_pair, query_features, key_features, output);
      break;
    case 64:
      attention_weight_computation_kernel<T, 64><<<blocks, threads, 0, stream>>>(
        b, total_query_num, local_size, total_key_num, nhead, hdim, query_batch_cnt, key_batch_cnt,
        index_pair_batch, index_pair, query_features, key_features, output);
      break;
    case 128:
      attention_weight_computation_kernel<T, 128><<<blocks, threads, 0, stream>>>(
        b, total_query_num, local_size, total_key_num, nhead, hdim, query_batch_cnt, key_batch_cnt,
        index_pair_batch, index_pair, query_features, key_features, output);
      break;
    default:
      attention_weight_computation_kernel<T, 150><<<blocks, threads, 0, stream>>>(
        b, total_query_num, local_size, total_key_num, nhead, hdim, query_batch_cnt, key_batch_cnt,
        index_pair_batch, index_pair, query_features, key_features, output);
      break;
  }
  return cudaGetLastError();
}

template cudaError_t AttentionWeightComputationLauncher<float>(
  const int32_t b, const int32_t total_query_num, const int32_t local_szie,
  const int32_t total_value_num, const int32_t nhead, const int32_t hdim,
  const int * query_batch_cnt, const int * key_batch_cnt, const int * index_pair_batch,
  const int * index_pair, const float * query_features, const float * key_features, float * output,
  cudaStream_t stream);
