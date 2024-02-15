#include "mtr/mtr.hpp"

#include "preprocess/agent_preprocess_kernel.cuh"

namespace mtr
{
TrtMTR::TrtMTR(
  const std::string & model_path, const std::string & precision, const MtrConfig & config,
  const BatchConfig & batch_config, const size_t max_workspace_size,
  const BuildConfig & build_config)
: config_(config)
{
  builder_ = std::make_unique<MTRBuilder>(
    model_path, precision, batch_config, max_workspace_size, build_config);
  builder_->setup();

  if (!builder_->isInitialized()) {
    return;
  }

  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
}

bool TrtMTR::doInference(AgentData & agent_data)
{
  initCudaPtr(agent_data);

  if (!preProcess(agent_data)) {
    return false;
  }
  return true;
}

void TrtMTR::initCudaPtr(AgentData & agent_data)
{
  // !!TODO!!

  // source data
  d_target_index_ = cuda::make_unique<int[]>(agent_data.TargetNum);
  d_label_index_ = cuda::make_unique<int[]>(agent_data.AgentNum);
  d_timestamps_ = cuda::make_unique<float[]>(agent_data.TimeLength);
  d_trajectory_ =
    cuda::make_unique<float[]>(agent_data.AgentNum * agent_data.TimeLength * agent_data.StateDim);

  // preprocessed input
  const size_t D = agent_data.StateDim + agent_data.ClassNum + agent_data.TimeLength + 3;
  d_in_trajectory_ = cuda::make_unique<float[]>(
    agent_data.TargetNum * agent_data.AgentNum * agent_data.TimeLength * D);
  d_in_trajectory_mask_ =
    cuda::make_unique<bool[]>(agent_data.TargetNum * agent_data.AgentNum * agent_data.TimeLength);
  d_in_last_pos_ = cuda::make_unique<float[]>(agent_data.TargetNum * agent_data.AgentNum * 3);

  // outputs
  d_out_scores_ = cuda::make_unique<float[]>(agent_data.TargetNum * config_.num_mode);
  d_out_trajectory_ = cuda::make_unique<float[]>(
    agent_data.TargetNum * config_.num_mode * config_.num_future);  // TODO output dimension

  // debug
  h_debug_in_trajectory_ = std::make_unique<float[]>(
    agent_data.TargetNum * agent_data.AgentNum * agent_data.TimeLength * D);
  h_debug_in_trajectory_mask_ =
    std::make_unique<bool[]>(agent_data.TargetNum * agent_data.AgentNum * agent_data.TimeLength);
  h_debug_in_last_pos_ = std::make_unique<float[]>(agent_data.TargetNum * agent_data.AgentNum * 3);
}

bool TrtMTR::preProcess(AgentData & agent_data)
{
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_target_index_.get(), agent_data.target_index.data(), sizeof(int) * agent_data.TargetNum,
    cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_label_index_.get(), agent_data.label_index.data(), sizeof(int) * agent_data.AgentNum,
    cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_timestamps_.get(), agent_data.timestamps.data(), sizeof(float) * agent_data.TimeLength,
    cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_trajectory_.get(), agent_data.data_ptr(),
    sizeof(float) * agent_data.AgentNum * agent_data.TimeLength * agent_data.StateDim,
    cudaMemcpyHostToDevice, stream_));

  // DEBUG
  event_debugger_.createEvent(stream_);
  // Preprocess
  CHECK_CUDA_ERROR(agentPreprocessLauncher(
    agent_data.TargetNum, agent_data.AgentNum, agent_data.TimeLength, agent_data.StateDim,
    agent_data.ClassNum, agent_data.sdc_index, d_target_index_.get(), d_label_index_.get(),
    d_timestamps_.get(), d_trajectory_.get(), d_in_trajectory_.get(), d_in_trajectory_mask_.get(),
    d_in_last_pos_.get(), stream_));
  event_debugger_.printElapsedTime(stream_);

  debugPreprocess(agent_data);

  return true;
}

void TrtMTR::debugPreprocess(const AgentData & agent_data)
{
  // DEBUG
  const size_t D = agent_data.StateDim + agent_data.ClassNum + agent_data.TimeLength + 3;
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    h_debug_in_trajectory_.get(), d_in_trajectory_.get(),
    sizeof(float) * agent_data.TargetNum * agent_data.AgentNum * agent_data.TimeLength * D,
    cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    h_debug_in_trajectory_mask_.get(), d_in_trajectory_mask_.get(),
    sizeof(bool) * agent_data.TargetNum * agent_data.AgentNum * agent_data.TimeLength,
    cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    h_debug_in_last_pos_.get(), d_in_last_pos_.get(),
    sizeof(float) * agent_data.TargetNum * agent_data.AgentNum * 3, cudaMemcpyDeviceToHost,
    stream_));

  std::cout << "=== Trajectory data ===\n";
  for (size_t b = 0; b < agent_data.TargetNum; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (size_t n = 0; n < agent_data.AgentNum; ++n) {
      std::cout << "  Agent " << n << ":\n";
      for (size_t t = 0; t < agent_data.TimeLength; ++t) {
        std::cout << "  Time " << t << ": ";
        for (size_t d = 0; d < D; ++d) {
          std::cout << h_debug_in_trajectory_.get()
                         [(b * agent_data.AgentNum * agent_data.TimeLength +
                           n * agent_data.TimeLength + t) *
                            D +
                          d]
                    << " ";
        }
        std::cout << "\n";
      }
    }
  }

  std::cout << "=== Trajectory mask ===\n";
  for (size_t b = 0; b < agent_data.TargetNum; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (size_t n = 0; n < agent_data.AgentNum; ++n) {
      std::cout << "  Agent " << n << ": ";
      for (size_t t = 0; t < agent_data.TimeLength; ++t) {
        std::cout << h_debug_in_trajectory_mask_
                       .get()[(b * agent_data.AgentNum + n) * agent_data.TimeLength + t]
                  << " ";
      }
      std::cout << "\n";
    }
  }

  std::cout << "=== Last pos ===\n";
  for (size_t b = 0; b < agent_data.TargetNum; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (size_t n = 0; n < agent_data.AgentNum; ++n) {
      std::cout << "  Agent " << n << ": ";
      for (size_t d = 0; d < 3; ++d) {
        std::cout << h_debug_in_last_pos_.get()[(b * agent_data.AgentNum + n) * 3 + d] << " ";
      }
      std::cout << "\n";
    }
  }
}
}  // namespace mtr