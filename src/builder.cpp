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

#include "mtr/builder.hpp"

#include <fstream>
#include <sstream>

namespace mtr
{
namespace
{
/**
 * @brief Get the name of precision in string.
 *
 * @param type
 * @return std::string
 */
std::string getPrecisionName(const PrecisionType & type)
{
  switch (type) {
    case PrecisionType::FP32:
      return "FP32";
    case PrecisionType::FP16:
      return "FP16";
    case PrecisionType::INT8:
      return "INT8";
    default:
      throw std::runtime_error("Unsupported precision type.");
  }
}

/**
 * @brief Get the name of calibration in string.
 *
 * @param type
 * @return std::string
 */
std::string getCalibrationName(const CalibrationType & type)
{
  switch (type) {
    case CalibrationType::ENTROPY:
      return "ENTROPY";
    case CalibrationType::LEGACY:
      return "LEGACY";
    case CalibrationType::PERCENTILE:
      return "PERCENTILE";
    case CalibrationType::MINMAX:
      return "MINMAX";
    default:
      throw std::runtime_error("Unsupported calibration type.");
  }
}
}  // namespace
MTRBuilder::MTRBuilder(
  const std::string & model_filepath, const BuildConfig & build_config,
  const BatchConfig & batch_config, const size_t max_workspace_size)
: model_filepath_(model_filepath),
  batch_config_(batch_config),
  max_workspace_size_(max_workspace_size)
{
  build_config_ = std::make_unique<const BuildConfig>(build_config);
  runtime_ = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
}

MTRBuilder::~MTRBuilder()
{
}

void MTRBuilder::setup()
{
  if (!fs::exists(model_filepath_)) {
    is_initialized_ = false;
    return;
  }
  std::string engine_path = model_filepath_;
  if (model_filepath_.extension() == ".engine") {
    std::cout << "Loading... " << model_filepath_ << std::endl;
    loadEngine(model_filepath_);
  } else if (model_filepath_.extension() == ".onnx") {
    const auto engine_cache_path = createEngineCachePath();
    if (fs::exists(engine_cache_path)) {
      std::cout << "Loading cached engine... " << engine_cache_path << std::endl;
      if (!loadEngine(engine_cache_path)) {
        std::cerr << "Fail to load engine" << std::endl;
        is_initialized_ = false;
        return;
      }
    } else {
      std::cout << "Building... " << engine_cache_path << std::endl;
      logger_.log(nvinfer1::ILogger::Severity::kINFO, "start build engine");
      if (!buildEngineFromOnnx(model_filepath_, engine_cache_path)) {
        std::cerr << "Fail to build engine from onnx" << std::endl;
        is_initialized_ = false;
        return;
      }
      logger_.log(nvinfer1::ILogger::Severity::kINFO, "End build engine");
    }
    engine_path = engine_cache_path;
  } else {
    is_initialized_ = false;
    return;
  }

  context_ = TrtUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  if (!context_) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create context");
    is_initialized_ = false;
    return;
  }

  is_initialized_ = true;
}

bool MTRBuilder::loadEngine(const std::string & filepath)
{
  try {
    std::ifstream engine_file(filepath);
    std::stringstream buffer;
    buffer << engine_file.rdbuf();
    std::string engine_str = buffer.str();
    engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(
      reinterpret_cast<const void *>(engine_str.data()), engine_str.size()));
    return true;
  } catch (std::exception & e) {
    std::cerr << e.what() << std::endl;
    return false;
  }
}

fs::path MTRBuilder::createEngineCachePath() const
{
  fs::path cache_engine_path{model_filepath_};
  std::string precision_name = getPrecisionName(build_config_->precision);
  std::string calibration_name = build_config_->precision == PrecisionType::INT8
                                   ? getCalibrationName(build_config_->calibration)
                                   : "";
  cache_engine_path.replace_extension(calibration_name + precision_name);
  return cache_engine_path;
}

bool MTRBuilder::buildEngineFromOnnx(
  const std::string & filepath, const std::string & output_engine_filepath)
{
  auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
  if (!builder) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create builder");
    return false;
  }

  const auto explicit_batch =
    1U << static_cast<int32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  auto network =
    TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
  if (!network) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create network");
    return false;
  }

  auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create builder config");
    return false;
  }

  if (build_config_->precision != PrecisionType::FP32) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
#if (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSOR_PATCH >= 8400
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, max_workspace_size_);
#else
  config->setMaxWorkspaceSize(max_workspace_size_);
#endif

  auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
  if (!parser->parseFromFile(
        filepath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR))) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to parse onnx file");
    return false;
  }

  const auto input0 = network->getInput(0);
  const auto input0_dims = input0->getDimensions();
  const auto num_targets = input0_dims.d[0];
  auto num_agents = input0_dims.d[1];
  const auto num_past_frames = input0_dims.d[2];
  const auto num_agent_dims = input0_dims.d[3];

  const auto input2 = network->getInput(2);
  const auto input2_dims = input2->getDimensions();
  const auto num_polylines = input2_dims.d[1];
  const auto num_points = input2_dims.d[2];
  const auto num_polyline_dims = input2_dims.d[3];

  // if (num_targets > 1) {
  //   batch_config_[0] = num_targets;
  // }

  if (num_agents == -1) {
    num_agents = 10;
  }

  std::cout << "batch_config: (" << batch_config_.at(0) << ", " << batch_config_.at(1) << ", "
            << batch_config_.at(2) << ")\n";

  if (build_config_->is_dynamic) {
    auto profile = builder->createOptimizationProfile();
    {  // trajectory
      auto name = network->getInput(0)->getName();
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{batch_config_.at(0), num_agents, num_past_frames, num_agent_dims});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{batch_config_.at(1), num_agents, num_past_frames, num_agent_dims});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{batch_config_.at(2), num_agents, num_past_frames, num_agent_dims});
    }
    {  // trajectory mask
      auto name = network->getInput(1)->getName();
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims3{batch_config_.at(0), num_agents, num_past_frames});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims3{batch_config_.at(1), num_agents, num_past_frames});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims3{batch_config_.at(2), num_agents, num_past_frames});
    }
    {  // polyline
      auto name = network->getInput(2)->getName();
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{batch_config_.at(0), num_polylines, num_points, num_polyline_dims});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{batch_config_.at(1), num_polylines, num_points, num_polyline_dims});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{batch_config_.at(2), num_polylines, num_points, num_polyline_dims});
    }
    {  // polyline mask
      auto name = network->getInput(3)->getName();
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims3{batch_config_.at(0), num_polylines, num_points});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims3{batch_config_.at(1), num_polylines, num_points});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims3{batch_config_.at(2), num_polylines, num_points});
    }
    {  // polyline center
      auto name = network->getInput(4)->getName();
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims3{batch_config_.at(0), num_polylines, 3});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims3{batch_config_.at(1), num_polylines, 3});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims3{batch_config_.at(2), num_polylines, 3});
    }
    {  // last pos
      auto name = network->getInput(5)->getName();
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims3{batch_config_.at(0), num_agents, 3});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims3{batch_config_.at(1), num_agents, 3});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims3{batch_config_.at(2), num_agents, 3});
    }
    {  // track index
      auto name = network->getInput(6)->getName();
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{batch_config_.at(0)});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{batch_config_.at(1)});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{batch_config_.at(2)});
    }
    {
      // intention points
      auto name = network->getInput(7)->getName();
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3{batch_config_.at(0), 64, 2});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3{batch_config_.at(1), 64, 2});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3{batch_config_.at(2), 64, 2});
    }
    {  // pred scores
      auto name = network->getOutput(0)->getName();
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{batch_config_.at(0), 6});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{batch_config_.at(1), 6});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{batch_config_.at(2), 6});
    }
    {  // pred trajs
      auto name = network->getOutput(1)->getName();
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{batch_config_.at(0), 6, 80, 7});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{batch_config_.at(1), 6, 80, 7});
      profile->setDimensions(
        name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{batch_config_.at(2), 6, 80, 7});
    }
    config->addOptimizationProfile(profile);
  }

  if (build_config_->is_dynamic) {
#if (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSOR_PATCH >= 8200
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
#else
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kVERBOSE);
#endif
  }

#if (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSOR_PATCH >= 8000
  auto plan =
    TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create host memory");
    return false;
  }
  engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(plan->data(), plan->size()));
#else
  engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
#endif

  if (!engine_) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create engine");
    return false;
  }

// save engine
#if TENSORRT_VERSION_MAJOR < 8
  auto data = TrtUniquePtr<nvinfer1::IHostMemory>(engine_->serialize());
#endif
  std::ofstream file;
  file.open(output_engine_filepath, std::ios::binary | std::ios::out);
  if (!file.is_open()) {
    return false;
  }
#if TENSORRT_VERSION_MAJOR < 8
  file.write(reinterpret_cast<const char *>(data->data()), data->size());
#else
  file.write(reinterpret_cast<const char *>(plan->data()), plan->size());
#endif

  file.close();

  return true;
}

bool MTRBuilder::isInitialized() const
{
  return is_initialized_;
}

bool MTRBuilder::isDynamic() const
{
  return build_config_->is_dynamic;
}

bool MTRBuilder::setBindingDimensions(int32_t index, nvinfer1::Dims dimensions)
{
  if (build_config_->is_dynamic) {
    return context_->setBindingDimensions(index, dimensions);
  } else {
    return true;
  }
}

bool MTRBuilder::enqueueV2(void ** bindings, cudaStream_t stream, cudaEvent_t * inputConsumed)
{
  return context_->enqueueV2(bindings, stream, inputConsumed);
}

}  // namespace mtr
