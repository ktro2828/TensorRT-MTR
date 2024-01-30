#include "common/trt_plugin_helper.hpp"
#include "knn/trt_knn_batch_kernel.hpp"

template <typename T>
__global__ void knn_batch_kernel(
  const int n, const int m, const int k, const T * xyz, const T * query_xyz, const int * batch_idxs,
  const int * query_batch_offsets, int * output)
{
  const int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pt_idx >= n) {
    return;
  }

  xyz += pt_idx * 3;
  output += pt_idx * k;

  T ox = xyz[0];
  T oy = xyz[1];
  T oz = xyz[2];

  T best[100];
  int best_idx[100];
  for (int i = 0; i < k; ++i) {
    best[i] = static_cast<T>(1e20);
    best_idx[i] = -1;
  }

  int batch_idx = batch_idxs[pt_idx];
  int start_idx = query_batch_offsets[batch_idx];
  int end_idx = query_batch_offsets[batch_idx + 1];
  for (int i = start_idx; i < end_idx; ++i) {
    T x = query_xyz[i * 3 + 0];
    T y = query_xyz[i * 3 + 1];
    T z = query_xyz[i * 3 + 2];
    T d2 = (ox - x) * (ox - x) + (oy - y) * (oy - y) + (oz - z) * (oz - z);
    for (int p = 0; p < k; ++p) {
      if (d2 < best[p]) {
        for (int q = k - 1; q > p; --q) {
          best[q] = best[q - 1];
          best_idx[q] = best_idx[q - 1];
        }
        best[p] = d2;
        best_idx[p] = i - start_idx;
        break;
      }
    }
  }

  for (int i = 0; i < k; ++i) {
    output[i] = best_idx[i];
  }
}

template <typename T>
cudaError_t KnnBatchLauncher(
  const int n, const int m, const int k, const T * xyz, const T * query_xyz, const int * batch_idxs,
  const int * query_batch_offsets, int * output, cudaStream_t stream)
{
  dim3 blocks(DIVUP(n, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK);

  knn_batch_kernel<T><<<blocks, threads, 0, stream>>>(
    n, m, k, xyz, query_xyz, batch_idxs, query_batch_offsets, output);

  return cudaGetLastError();
}

template cudaError_t KnnBatchLauncher<float>(
  const int n, const int m, const int k, const float * xyz, const float * query_xyz,
  const int * batch_idxs, const int * query_batch_offsets, int * output, cudaStream_t stream);
