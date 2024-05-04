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
  const std::string & model_path, const MTRConfig & config, const BuildConfig & build_config,
  const size_t max_workspace_size)
: config_(config),
  intention_point_(config_.intention_point_filepath, config_.num_intention_point_cluster)
{
  builder_ = std::make_unique<MTRBuilder>(model_path, build_config, max_workspace_size);
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
  num_target_ = agent_data.num_target();
  num_agent_ = agent_data.num_agent();
  num_timestamp_ = agent_data.num_timestamp();
  num_agent_dim_ = agent_data.num_state_dim();
  num_agent_class_ = agent_data.num_class();
  num_agent_attr_ = agent_data.num_attr();
  num_polyline_ = polyline_data.num_polyline();
  num_point_ = polyline_data.num_point();
  num_point_dim_ = polyline_data.num_state_dim();
  num_point_attr_ = polyline_data.num_attr();

  // source data
  d_target_index_ = cuda::make_unique<int[]>(num_target_);
  d_label_index_ = cuda::make_unique<int[]>(num_agent_);
  d_timestamps_ = cuda::make_unique<float[]>(num_timestamp_);
  d_trajectory_ = cuda::make_unique<float[]>(agent_data.size());
  d_target_state_ = cuda::make_unique<float[]>(num_target_ * num_agent_dim_);
  d_intention_points_ =
    cuda::make_unique<float[]>(num_agent_ * config_.num_intention_point_cluster * 2);
  d_polyline_ = cuda::make_unique<float[]>(polyline_data.size());

  // preprocessed input
  d_in_trajectory_ = cuda::make_unique<float[]>(agent_data.input_size());
  d_in_trajectory_mask_ = cuda::make_unique<bool[]>(num_target_ * num_agent_ * num_timestamp_);
  d_in_last_pos_ = cuda::make_unique<float[]>(num_target_ * num_agent_ * 3);
  d_in_polyline_ = cuda::make_unique<float[]>(
    num_target_ * config_.max_num_polyline * num_point_ * num_point_attr_);
  d_in_polyline_mask_ =
    cuda::make_unique<bool[]>(num_target_ * config_.max_num_polyline * num_point_);
  d_in_polyline_center_ = cuda::make_unique<float[]>(num_target_ * config_.max_num_polyline * 3);

  if (config_.max_num_polyline < num_polyline_) {
    d_tmp_polyline_ =
      cuda::make_unique<float[]>(num_target_ * num_polyline_ * num_point_ * num_point_attr_);
    d_tmp_polyline_mask_ = cuda::make_unique<bool[]>(num_target_ * num_polyline_ * num_point_);
    d_tmp_distance_ = cuda::make_unique<float[]>(num_target_ * num_polyline_);
  }

  if (builder_->isDynamic()) {
    // TODO(ktro2828): refactor
    // obj_trajs
    builder_->setBindingDimensions(
      0, nvinfer1::Dims4{num_target_, num_agent_, num_timestamp_, num_agent_attr_});
    // obj_trajs_mask
    builder_->setBindingDimensions(1, nvinfer1::Dims3{num_target_, num_agent_, num_timestamp_});
    // polylines
    builder_->setBindingDimensions(
      2, nvinfer1::Dims4{num_target_, config_.max_num_polyline, num_point_, num_point_attr_});
    // polyline mask
    builder_->setBindingDimensions(
      3, nvinfer1::Dims3{num_target_, config_.max_num_polyline, num_point_});
    // polyline center
    builder_->setBindingDimensions(4, nvinfer1::Dims3{num_target_, config_.max_num_polyline, 3});
    // obj last pos
    builder_->setBindingDimensions(5, nvinfer1::Dims3{num_target_, num_agent_, 3});
    // track index to predict
    nvinfer1::Dims targetIdxDim;
    targetIdxDim.nbDims = 1;
    targetIdxDim.d[0] = num_target_;
    builder_->setBindingDimensions(6, targetIdxDim);
    // intention points
    builder_->setBindingDimensions(
      7, nvinfer1::Dims3{num_target_, config_.num_intention_point_cluster, 2});
  }

  // outputs
  d_out_score_ = cuda::make_unique<float[]>(num_target_ * config_.num_mode);
  d_out_trajectory_ = cuda::make_unique<float[]>(
    num_target_ * config_.num_mode * config_.num_future * PredictedStateDim);
}

bool TrtMTR::preProcess(const AgentData & agent_data, const PolylineData & polyline_data)
{
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_target_index_.get(), agent_data.target_indices().data(), sizeof(int) * num_target_,
    cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_label_index_.get(), agent_data.label_indices().data(), sizeof(int) * num_agent_,
    cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_timestamps_.get(), agent_data.timestamps().data(), sizeof(float) * num_timestamp_,
    cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_trajectory_.get(), agent_data.data_ptr(), sizeof(float) * agent_data.size(),
    cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_target_state_.get(), agent_data.target_data_ptr(),
    sizeof(float) * num_target_ * num_agent_dim_, cudaMemcpyHostToDevice, stream_));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_polyline_.get(), polyline_data.data_ptr(), sizeof(float) * polyline_data.size(),
    cudaMemcpyHostToDevice, stream_));

  const auto target_label_names = getLabelNames(agent_data.target_label_indices());
  const auto intention_points = intention_point_.get_points(target_label_names);
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_intention_points_.get(), intention_points.data(),
    sizeof(float) * num_target_ * config_.num_intention_point_cluster * 2, cudaMemcpyHostToDevice,
    stream_));

  // DEBUG
  event_debugger_.createEvent(stream_);

  // Preprocess
  CHECK_CUDA_ERROR(agentPreprocessLauncher(
    num_target_, num_agent_, num_timestamp_, num_agent_dim_, num_agent_class_,
    agent_data.ego_index(), d_target_index_.get(), d_label_index_.get(), d_timestamps_.get(),
    d_trajectory_.get(), d_in_trajectory_.get(), d_in_trajectory_mask_.get(), d_in_last_pos_.get(),
    stream_));

  if (config_.max_num_polyline < num_polyline_) {
    CHECK_CUDA_ERROR(polylinePreprocessWithTopkLauncher(
      config_.max_num_polyline, num_polyline_, num_point_, num_point_dim_, d_polyline_.get(),
      num_target_, num_agent_dim_, d_target_state_.get(), d_tmp_polyline_.get(),
      d_tmp_polyline_mask_.get(), d_tmp_distance_.get(), d_in_polyline_.get(),
      d_in_polyline_mask_.get(), d_in_polyline_center_.get(), stream_));
  } else {
    CHECK_CUDA_ERROR(polylinePreprocessLauncher(
      num_polyline_, num_point_, num_point_dim_, d_polyline_.get(), num_target_, num_agent_dim_,
      d_target_state_.get(), d_in_polyline_.get(), d_in_polyline_mask_.get(),
      d_in_polyline_center_.get(), stream_));
  }

  event_debugger_.printElapsedTime(stream_);

  return true;
}

bool TrtMTR::postProcess(
  const AgentData & agent_data, std::vector<PredictedTrajectory> & trajectories)
{
  // Postprocess
  CHECK_CUDA_ERROR(postprocessLauncher(
    num_target_, config_.num_mode, config_.num_future, num_agent_dim_, d_target_state_.get(),
    PredictedStateDim, d_out_trajectory_.get(), stream_));

  // clear containers on the host device and reserve size for the allocation.
  h_out_score_.clear();
  h_out_trajectory_.clear();
  h_out_score_.reserve(num_target_ * config_.num_mode);
  h_out_trajectory_.reserve(
    num_target_ * config_.num_mode * config_.num_future * PredictedStateDim);

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    h_out_score_.data(), d_out_score_.get(), sizeof(float) * num_target_ * config_.num_mode,
    cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    h_out_trajectory_.data(), d_out_trajectory_.get(),
    sizeof(float) * num_target_ * config_.num_mode * config_.num_future * PredictedStateDim,
    cudaMemcpyDeviceToHost, stream_));

  trajectories.reserve(num_target_);
  for (size_t b = 0; b < num_target_; ++b) {
    const auto score_itr = h_out_score_.cbegin() + config_.num_mode;
    std::vector<float> scores(score_itr, score_itr + config_.num_mode);
    const auto mode_itr =
      h_out_trajectory_.cbegin() + b * config_.num_mode * config_.num_future * PredictedStateDim;
    std::vector<float> modes(
      mode_itr, mode_itr + config_.num_mode * config_.num_future * PredictedStateDim);
    trajectories.emplace_back(scores, modes, config_.num_mode, config_.num_future);
  }

  return true;
}
}  // namespace mtr
