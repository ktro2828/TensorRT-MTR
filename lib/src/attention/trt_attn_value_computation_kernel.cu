#include "attention/trt_attn_value_computation_kernel.hpp"

template <typename T, unsigned int d>
__global__ void attention_value_computation_kernel(
  int b, int total_query_num, int local_size, int total_value_num, int nhead, int hdim,
  const int * query_batch_cnt, const int * key_batch_cnt, const int * index_pair_batch,
  const int * index_pair, const T * attn_weight, const T * value_features, T * output)
{
  const int query_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int hdim_idx = threadIdx.x;

  if (query_idx >= total_query_num || head_idx >= nhead || hdim_idx >= hdim) {
    return;
  }

  // get key_start_idx
  const int batch_idx = index_pair_batch[query_idx];
  int key_start_idx = 0;
  for (int i = 0; i < batch_idx; ++i) {
    key_start_idx += key_batch_cnt[i];
  }
  // get shared variables
  __shared__ T shared_attn_weight[d];
  __shared__ int shared_value_indices[d];
  int cur_key_idx = 0;
  for (int i = 0; i < local_size; i += blockDim.x) {
    shared_attn_weight[i] = attn_weight[query_idx * local_size * nhead + i * nhead + head_idx];

    cur_key_idx = index_pair[query_idx * local_size + i];
    if (cur_key_idx == -1) {
      shared_value_indices[i] = -1;
      continue;
    }
    cur_key_idx += key_start_idx;
    shared_value_indices[i] = cur_key_idx;
  }
  __syncthreads();

  output += query_idx * nhead * hdim + head_idx * hdim + hdim_idx;

  T attn_result = 0;
  for (int i = 0; i < local_size; ++i) {
    if (shared_value_indices[i] == -1) {
      attn_result +=
        shared_attn_weight[i] *
        value_features[shared_value_indices[i] * nhead * hdim + head_idx * hdim + hdim_idx];
    }
  }
  output[0] = attn_result;
}

template <typename T>
void AttentionValueComputationLauncher(
  int b, int total_query_num, int local_size, int total_value_num, int nhead, int hdim,
  const int * query_batch_cnt, const int * key_batch_cnt, const int * index_pair_batch,
  const int * index_pair, const T * attn_weight, const T * value_features, T * output,
  cudaStream_t stream)
{
  if (local_size > 512) {
    // TODO: WARNING
    return;
  }

  dim3 blocks(total_query_num, nhead);
  dim3 threads(hdim);

  switch (local_size) {
    case 16:
      attention_value_computation_kernel<T, 16><<<blocks, threads, 0, stream>>>(
        b, total_query_num, local_size, total_value_num, nhead, hdim, query_batch_cnt,
        key_batch_cnt, index_pair_batch, index_pair, attn_weight, value_features, output);
      break;
    case 32:
      attention_value_computation_kernel<T, 32><<<blocks, threads, 0, stream>>>(
        b, total_query_num, local_size, total_value_num, nhead, hdim, query_batch_cnt,
        key_batch_cnt, index_pair_batch, index_pair, attn_weight, value_features, output);
      break;
    case 64:
      attention_value_computation_kernel<T, 64><<<blocks, threads, 0, stream>>>(
        b, total_query_num, local_size, total_value_num, nhead, hdim, query_batch_cnt,
        key_batch_cnt, index_pair_batch, index_pair, attn_weight, value_features, output);
      break;
    case 128:
      attention_value_computation_kernel<T, 128><<<blocks, threads, 0, stream>>>(
        b, total_query_num, local_size, total_value_num, nhead, hdim, query_batch_cnt,
        key_batch_cnt, index_pair_batch, index_pair, attn_weight, value_features, output);
      break;
    case 320:
      attention_value_computation_kernel<T, 320><<<blocks, threads, 0, stream>>>(
        b, total_query_num, local_size, total_value_num, nhead, hdim, query_batch_cnt,
        key_batch_cnt, index_pair_batch, index_pair, attn_weight, value_features, output);
      break;
    case 384:
      attention_value_computation_kernel<T, 384><<<blocks, threads, 0, stream>>>(
        b, total_query_num, local_size, total_value_num, nhead, hdim, query_batch_cnt,
        key_batch_cnt, index_pair_batch, index_pair, attn_weight, value_features, output);
      break;
    default:
      attention_value_computation_kernel<T, 512><<<blocks, threads, 0, stream>>>(
        b, total_query_num, local_size, total_value_num, nhead, hdim, query_batch_cnt,
        key_batch_cnt, index_pair_batch, index_pair, attn_weight, value_features, output);
      break;
  }
}