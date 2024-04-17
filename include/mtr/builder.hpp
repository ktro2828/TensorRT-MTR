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

#ifndef MTR__BUILDER_HPP_
#define MTR__BUILDER_HPP_

#include "mtr/logger.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <filesystem>
namespace fs = ::std::filesystem;

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace mtr
{

template <typename T>
struct TrtDeleter
{
  void operator()(T * obj) const
  {
    if (obj) {
#if NV_TENSORRT_MAJOR >= 8
      delete obj;
#else
      obj->destroy();
#endif
    }
  }
};  // struct TrtDeleter

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;
using BatchConfig = std::array<int32_t, 3>;

enum PrecisionType { FP32 = 0, FP16 = 1, INT8 = 2 };
enum CalibrationType { ENTROPY = 0, LEGACY = 1, PERCENTILE = 2, MINMAX = 3 };

struct BuildConfig
{
  // type of precision
  PrecisionType precision;

  // type for calibration
  CalibrationType calibration;

  // flag of dynamic shape
  bool is_dynamic;

  /**
   * @brief Construct a new instance with default configurations.
   */
  BuildConfig(const bool is_dynamic = false)
  : precision(PrecisionType::FP32), calibration(CalibrationType::MINMAX), is_dynamic(is_dynamic)
  {
  }

  /**
   * @brief Construct a new build config.
   *
   * @param precision
   * @param calibration
   * @param is_dynamic
   */
  BuildConfig(
    const PrecisionType & precision, const CalibrationType & calibration,
    const bool is_dynamic = false)
  : precision(precision), calibration(calibration), is_dynamic(is_dynamic)
  {
  }
};  // struct BuildConfig

class MTRBuilder
{
public:
  /**
   * @brief Construct a new instance.
   *
   * @param model_path Path to engine or onnx file.
   * @param build_config The configuration of build.
   * @param batch_config The configuration of min/opt/max batch.
   * @param max_workspace_size The max workspace size.
   */
  MTRBuilder(
    const std::string & model_path, const BuildConfig & build_config = BuildConfig(),
    const BatchConfig & batch_config = {1, 1, 1}, const size_t max_workspace_size = (1ULL << 30));

  /**
   * @brief Destroy the instance.
   */
  ~MTRBuilder();

  /**
   * @brief Setup engine for inference. After finishing setup successfully, `isInitialized` must
   * return `true`.
   */
  void setup();

  /**
   * @brief Check whether engine was initialized successfully.
   *
   * @return True if plugins were initialized successfully.
   */
  bool isInitialized() const;

  bool isDynamic() const;

  bool setBindingDimensions(int32_t index, nvinfer1::Dims dimensions);

  /**
   * @brief A wrapper of `nvinfer1::IExecuteContext::enqueueV2`.
   *
   * @param bindings An array of pointers to input and output buffers for the network.
   * @param stream A cuda stream on which the inference kernels will be enqueued.
   * @param inputConsumed An optional event which will be signaled when the input buffers can be
   * refilled with new data.
   * @return True If the kernels were enqueued successfully.
   */
  bool enqueueV2(void ** bindings, cudaStream_t stream, cudaEvent_t * inputConsumed);

private:
  /**
   * @brief Load engin file.
   *
   * @param filepath Engine file path.
   * @return True if the engine were loaded successfully.
   */
  bool loadEngine(const std::string & filepath);

  /**
   * @brief Create a cache path of engine file.
   *
   * @return fs::path
   */
  fs::path createEngineCachePath() const;

  /**
   * @brief Build engine from onnx file.
   *
   * @param filepath Onnx file path.
   * @param output_engine_filepath Output engine file path.
   * @return True if the engine were built successfully.
   */
  bool buildEngineFromOnnx(
    const std::string & filepath, const std::string & output_engine_filepath);

  Logger logger_;
  TrtUniquePtr<nvinfer1::IRuntime> runtime_;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
  TrtUniquePtr<nvinfer1::IExecutionContext> context_;

  fs::path model_filepath_;
  BatchConfig batch_config_;
  size_t max_workspace_size_;
  std::unique_ptr<const BuildConfig> build_config_;

  bool is_initialized_{false};
};  // class MTRBuilder
}  // namespace mtr
#endif  // MTR__BUILDER_HPP_
