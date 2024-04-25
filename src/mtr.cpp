// Copyright 2024 Kotaro Uetake
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mtr/mtr.hpp"

#include "postprocess/postprocess_kernel.cuh"
#include "preprocess/agent_preprocess_kernel.cuh"
#include "preprocess/polyline_preprocess_kernel.cuh"

namespace mtr
{
TrtMTR::TrtMTR(
  const std::string & model_path, const std::string & precision, const MTRConfig & config,
  const BatchConfig & batch_config, const size_t max_workspace_size,
  const BuildConfig & build_config)
: config_(config),
  intention_point_(config_.intention_point_filepath, config_.num_intention_point_cluster)
{
  builder_ = std::make_unique<MTRBuilder>(
    model_path, precision, batch_config, max_workspace_size, build_config);
  builder_->setup();

  if (!builder_->isInitialized()) {
    return;
  }

  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
}

bool TrtMTR::doInference(
  const AgentData & agent_data, const PolylineData & polyline_data,
  std::vector<PredictedTrajectory> & trajectories)
{
  initCudaPtr(agent_data, polyline_data);

  if (!preProcess(agent_data, polyline_data)) {
    std::cerr << "Fail to preprocess" << std::endl;
    return false;
  }

  std::vector<void *> buffer = {d_in_trajectory_.get(),      d_in_trajectory_mask_.get(),
                                d_in_polyline_.get(),        d_in_polyline_mask_.get(),
                                d_in_polyline_center_.get(), d_in_last_pos_.get(),
                                d_target_index_.get(),       d_intention_points_.get(),
                                d_out_score_.get(),          d_out_trajectory_.get()};

  if (!builder_->enqueueV2(buffer.data(), stream_, nullptr)) {
    std::cerr << "Fail to do inference" << std::endl;
    return false;
  };

  if (!postProcess(agent_data, trajectories)) {
    std::cerr << "Fail to preprocess" << std::endl;
    return false;
  }

  return true;
}

void TrtMTR::initCudaPtr(const AgentData & agent_data, const PolylineData & polyline_data)
{
  // source data
  d_target_index_ = cuda::make_unique<int[]>(agent_data.TargetNum);
  d_label_index_ = cuda::make_unique<int[]>(agent_data.AgentNum);
  d_timestamps_ = cuda::make_unique<float[]>(agent_data.TimeLength);
  d_trajectory_ =
    cuda::make_unique<float[]>(agent_data.AgentNum * agent_data.TimeLength * agent_data.StateDim);
  d_target_state_ = cuda::make_unique<float[]>(agent_data.TargetNum * agent_data.StateDim);
  d_intention_points_ =
    cuda::make_unique<float[]>(agent_data.TargetNum * config_.num_intention_point_cluster * 2);
  d_polyline_ = cuda::make_unique<float[]>(
    polyline_data.PolylineNum * polyline_data.PointNum * polyline_data.StateDim);

  // preprocessed input
  const size_t inAgentDim = agent_data.StateDim + agent_data.ClassNum + agent_data.TimeLength +
                            3;  // TODO: define this in global
  const size_t inPointDim = polyline_data.StateDim + 2;
  d_in_trajectory_ = cuda::make_unique<float[]>(
    agent_data.TargetNum * agent_data.AgentNum * agent_data.TimeLength * inAgentDim);
  d_in_trajectory_mask_ =
    cuda::make_unique<bool[]>(agent_data.TargetNum * agent_data.AgentNum * agent_data.TimeLength);
  d_in_last_pos_ = cuda::make_unique<float[]>(agent_data.TargetNum * agent_data.AgentNum * 3);
  d_in_polyline_ = cuda::make_unique<float[]>(
    agent_data.TargetNum * config_.max_num_polyline * polyline_data.PointNum * inPointDim);
  d_in_polyline_mask_ = cuda::make_unique<bool[]>(
    agent_data.TargetNum * config_.max_num_polyline * polyline_data.PointNum);
  d_in_polyline_center_ =
    cuda::make_unique<float[]>(agent_data.TargetNum * config_.max_num_polyline * 3);

  if (config_.max_num_polyline < polyline_data.PolylineNum) {
    d_tmp_polyline_ = cuda::make_unique<float[]>(
      agent_data.TargetNum * polyline_data.PolylineNum * polyline_data.PointNum * inPointDim);
    d_tmp_polyline_mask_ = cuda::make_unique<bool[]>(
      agent_data.TargetNum * polyline_data.PolylineNum * polyline_data.PointNum);
    d_tmp_distance_ = cuda::make_unique<float[]>(agent_data.TargetNum * polyline_data.PolylineNum);
  }

  // outputs
  d_out_score_ = cuda::make_unique<float[]>(agent_data.TargetNum * config_.num_mode);
  d_out_trajectory_ = cuda::make_unique<float[]>(
    agent_data.TargetNum * config_.num_mode * config_.num_future * PredictedStateDim);
  h_out_score_ = std::make_unique<float[]>(sizeof(float) * agent_data.TargetNum * config_.num_mode);
  h_out_trajectory_ = std::make_unique<float[]>(
    sizeof(float) * agent_data.TargetNum * config_.num_mode * config_.num_future *
    PredictedStateDim);
}

bool TrtMTR::preProcess(const AgentData & agent_data, const PolylineData & polyline_data)
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
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_target_state_.get(), agent_data.target_data_ptr(),
    sizeof(float) * agent_data.TargetNum * agent_data.StateDim, cudaMemcpyHostToDevice, stream_));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_polyline_.get(), polyline_data.data_ptr(),
    sizeof(float) * polyline_data.PolylineNum * polyline_data.PointNum * polyline_data.StateDim,
    cudaMemcpyHostToDevice, stream_));

  const auto target_label_names = getLabelNames(agent_data.target_label_index);
  const auto intention_points = intention_point_.get_points(target_label_names);
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_intention_points_.get(), intention_points.data(),
    sizeof(float) * agent_data.TargetNum * config_.num_intention_point_cluster * 2,
    cudaMemcpyHostToDevice, stream_));

  // DEBUG
  event_debugger_.createEvent(stream_);

  // Preprocess
  CHECK_CUDA_ERROR(agentPreprocessLauncher(
    agent_data.TargetNum, agent_data.AgentNum, agent_data.TimeLength, agent_data.StateDim,
    agent_data.ClassNum, agent_data.sdc_index, d_target_index_.get(), d_label_index_.get(),
    d_timestamps_.get(), d_trajectory_.get(), d_in_trajectory_.get(), d_in_trajectory_mask_.get(),
    d_in_last_pos_.get(), stream_));

  if (config_.max_num_polyline < polyline_data.PolylineNum) {
    CHECK_CUDA_ERROR(polylinePreprocessWithTopkLauncher(
      config_.max_num_polyline, polyline_data.PolylineNum, polyline_data.PointNum,
      polyline_data.StateDim, d_polyline_.get(), agent_data.TargetNum, agent_data.StateDim,
      d_target_state_.get(), d_tmp_polyline_.get(), d_tmp_polyline_mask_.get(),
      d_tmp_distance_.get(), d_in_polyline_.get(), d_in_polyline_mask_.get(),
      d_in_polyline_center_.get(), stream_));
  } else {
    assert(
      ("The number of config.max_num_polyline and PolylineData.PolylineNum must be same",
       config_.max_num_polyline == polyline_data.PolylineNum));
    CHECK_CUDA_ERROR(polylinePreprocessLauncher(
      polyline_data.PolylineNum, polyline_data.PointNum, polyline_data.StateDim, d_polyline_.get(),
      agent_data.TargetNum, agent_data.StateDim, d_target_state_.get(), d_in_polyline_.get(),
      d_in_polyline_mask_.get(), d_in_polyline_center_.get(), stream_));
  }

  event_debugger_.printElapsedTime(stream_);

  return true;
}

bool TrtMTR::postProcess(
  const AgentData & agent_data, std::vector<PredictedTrajectory> & trajectories)
{
  // Postprocess
  CHECK_CUDA_ERROR(postprocessLauncher(
    agent_data.TargetNum, config_.num_mode, config_.num_future, agent_data.StateDim,
    d_target_state_.get(), PredictedStateDim, d_out_trajectory_.get(), stream_));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    h_out_score_.get(), d_out_score_.get(), sizeof(float) * agent_data.TargetNum * config_.num_mode,
    cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    h_out_trajectory_.get(), d_out_trajectory_.get(),
    sizeof(float) * agent_data.TargetNum * config_.num_mode * config_.num_future *
      PredictedStateDim,
    cudaMemcpyDeviceToHost, stream_));

  trajectories.reserve(agent_data.TargetNum);
  for (size_t b = 0; b < agent_data.TargetNum; ++b) {
    const auto score_ptr = h_out_score_.get() + b * config_.num_mode;
    const auto trajectory_ptr =
      h_out_trajectory_.get() + b * config_.num_mode * config_.num_future * PredictedStateDim;
    trajectories.emplace_back(score_ptr, trajectory_ptr, config_.num_mode, config_.num_future);
  }

  return true;
}
}  // namespace mtr
