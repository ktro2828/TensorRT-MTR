#ifndef POSTPROCESS__POSTPROCESS_KERNEL_CUH__
#define POSTPROCESS__POSTPROCESS_KERNEL_CUH__

#include <cuda_runtime.h>

/**
 * @brief Execute postprocess to predicted score and trajectory.
 *
 * @param B The number of target agents.
 * @param M The number of modes.
 * @param inTime The number of past timestamps.
 * @param inDim The number of input agent state dimensions.
 * @param outTime The number of future timestamps.
 * @param outDim The number predicted agent state dimensions
 * @param target_state Target agent states at the latest timestamp, in shape [B, inDim].
 * @param pred_scores Predicted scores, in shape [B*M].
 * @param pred_trajectory Predicted trajectories, in shape [B*M*T*D].
 * @param stream CUDA stream.
 * @return cudaError_t
 */
cudaError_t postprocessLauncher(
  const int B, const int M, const int inTime, const int inDim, const int outTime, const int outDim,
  const float * target_state, float * pred_score, float * pred_trajectory, cudaStream_t stream);

#endif  // POSTPROCESS__POSTPROCESS_KERNEL_CUH__