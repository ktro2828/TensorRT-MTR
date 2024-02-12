#ifndef PREPROCESS__AGENT_PREPROCESS_KERNEL_HPP_
#define PREPROCESS__AGENT_PREPROCESS_KERNEL_HPP_

/**
 * @brief Transform trajectory to target agents centric coordinates system.
 *
 * @param B The number of targets.
 * @param N The number of agents.
 * @param T The number timstamps.
 * @param D The number of agent state dimensions.
 * @param target_index The array of target indices, in shape [B].
 * @param in_trajectory The array of trajectory, in shape [N*T*D].
 * @param output Output trajectoory, in shape [B*N*T*D].
 */
__device__ void transform_trajectory(
  const int B, const int N, const int T, const int D, const int * target_index,
  const float * in_trajectory, float * output);

/**
 * @brief Generate onehot mask in shape [B*N*T*(C+2)].
 *
 * @param B The number of targets.
 * @param N The number of agents.
 * @param T The number of timestamps.
 * @param C The number of agent classes.
 * @param sdc_index Ego index.
 * @param target_index Indices of target agents, in shape [B].
 * @param object_type_index Indices of classess of target agents, in shape [N].
 * @param onehot Output onehot mask, in shape [B*N*T*(C+2)]
 */
__device__ void generate_onehot_mask(
  const int B, const int N, const int T, const int C, const int sdc_index, const int * target_index,
  const int * object_type_index, float * onehot);

/**
 * @brief Generate embeddings for timestamps and headings.
 *
 * @param B The number of targets.
 * @param N The number of agents.
 * @param T The number of timestamps.
 * @param D The number of agent state dimensions.
 * @param timestamps The array of timestamps, in shape [T].
 * @param trajectory The array of trajectory, in shape [B*N*T*D].
 * @param time_embed Output timestamp embedding, in shape [B*N*T*(T+1)].
 * @param heading_embed Output heading embedding, in shape [B*N*T*2], ordering (sin, cos).
 */
__device__ void generate_embedding(
  const int B, const int N, const int T, const int D, const float * timestamps,
  const float * trajectory, float * time_embed, float * heading_embed);

/**
 * @brief Extract last position from the past history.
 *
 * @param B The number of targets.
 * @param N The number of agents.
 * @param T The number of timestamps.
 * @param D The number of agent state dimensions.
 * @param trajectory The array of trajectory, in shape [B*N*T*D].
 * @param output Output array of last positions, in shape [B*N*3].
 */
__device__ void extract_last_pos(
  const int B, const int N, const int T, const int D, const float * trajectory, float * output);

/**
 * @brief Concatenate transformed trajectory, onehot mask and embeddings.
 *
 * @param B The number of targets.
 * @param N The number of agents.
 * @param T The number of timestamps.
 * @param D The number of agent state dimensions.
 * @param C The number of agent classes.
 * @param trajectory The array of trajectory, in shape [B*N*T*D].
 * @param onehot The array of onehot mask, in shape [B*N*T*(C+2)].
 * @param time_embed The array of timestamp embedding, in shape [B*N*T*(T+1)].
 * @param heading_embed The array of heding embedding, in shape [B*N*T*2].
 * @param out_data Output concatenated data.
 * @param out_mask Output mask of concatenated data.
 */
__device__ void concatenate_agent_data(
  const int B, const int N, const int T, const int D, const int C, const float * trajectory,
  const float * onehot, const float * time_embed, const float * heading_embed, float * out_data,
  bool * out_mask);

/**
 * @brief
 *
 * @param B
 * @param N
 * @param T
 * @param D
 * @param C
 * @param sdc_index
 * @param target_index
 * @param object_type_index
 * @param timestamps
 * @param in_trajectory
 * @param out_data
 * @param out_mask
 * @param out_last_pos
 * @return __global__
 */
__global__ void agentPreprocessKernel(
  const int B, const int N, const int T, const int D, const int C, const int sdc_index,
  const int * target_index, const int * object_type_index, const float * timestamps,
  const float * in_trajectory, float * out_data, bool * out_mask, float * out_last_pos);

#endif  // PREPROCESS__AGENT_PREPROCESS_KERNEL_HPP_