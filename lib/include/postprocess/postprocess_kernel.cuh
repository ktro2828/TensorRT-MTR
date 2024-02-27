#ifndef POSTPROCESS__POSTPROCESS_KERNEL_CUH__
#define POSTPROCESS__POSTPROCESS_KERNEL_CUH__

#include <cuda_runtime.h>

/**
 * @brief A kernel to transform predicted trajectory from each target coords system to world coords
 * system.
 *
 * @param B The number of target agents.
 * @param M The number of modes.
 * @param T The number of future timestmaps.
 * @param inDim The number of input agent state dimensions.
 * @param targetState Source target agent states at latest timestamp, in shape [B*inDim].
 * @param outDim The number of output state dimensions.
 * @param trajectory Output predicted trajectory, in shape [B*M*T*outDim].
 * @return __global__
 */
__global__ void transformTrajectoryKernel(
  const int B, const int M, const int T, const int inDim, const float * targetState,
  const int outDim, float * trajectory);

/**
 * @brief Execute postprocess to predicted score and trajectory.
 *
 * @param B The number of target agents.
 * @param M The number of modes.
 * @param T The number of future timestamps.
 * @param inDim The number of input agent state dimensions.
 * @param target_state Target agent states at the latest timestamp, in shape [B * inDim].
 * @param outDim The number predicted agent state dimensions
 * @param pred_scores Predicted scores, in shape [B*M].
 * @param pred_trajectory Predicted trajectories, in shape [B*M*T*D].
 * @param stream CUDA stream.
 * @return cudaError_t
 */
cudaError_t postprocessLauncher(
  const int B, const int M, const int T, const int inDim, const float * target_state,
  const int outDim, float * pred_score, float * pred_trajectory, cudaStream_t stream);

#endif  // POSTPROCESS__POSTPROCESS_KERNEL_CUH__