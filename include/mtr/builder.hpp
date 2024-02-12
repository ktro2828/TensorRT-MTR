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

struct BuildConfig
{
  // type for calibration
  std::string calib_type_str;

  // DLA core ID that the process uses
  int dla_core_id;

  // flag for partial quantization in first layer
  bool quantize_first_layer;  // For partial quantization

  // flag for partial quantization in last layer
  bool quantize_last_layer;  // For partial quantization

  // flag for per-layer profiler using IProfiler
  bool profile_per_layer;

  // clip value for implicit quantization
  double clip_value;  // For implicit quantization

  // Supported calibration type
  const std::array<std::string, 4> valid_calib_type = {"Entropy", "Legacy", "Percentile", "MinMax"};

  BuildConfig()
  : calib_type_str("MinMax"),
    dla_core_id(-1),
    quantize_first_layer(false),
    quantize_last_layer(false),
    profile_per_layer(false),
    clip_value(0.0)
  {
  }

  explicit BuildConfig(
    const std::string & calib_type_str, const int dla_core_id = -1,
    const bool quantize_first_layer = false, const bool quantize_last_layer = false,
    const bool profile_per_layer = false, const double clip_value = 0.0)
  : calib_type_str(calib_type_str),
    dla_core_id(dla_core_id),
    quantize_first_layer(quantize_first_layer),
    quantize_last_layer(quantize_last_layer),
    profile_per_layer(profile_per_layer),
    clip_value(clip_value)
  {
    if (
      std::find(valid_calib_type.begin(), valid_calib_type.end(), calib_type_str) ==
      valid_calib_type.end()) {
      std::stringstream message;
      message << "Invalid calibration type was specified: " << calib_type_str << std::endl
              << "Valid value is one of: [Entropy, (Legacy | Percentile), MinMax]" << std::endl
              << "Default calibration type will be used: MinMax" << std::endl;
      std::cerr << message.str();
    }
  }
};  // struct BuildConfig

class MTRBuilder
{
public:
  MTRBuilder(
    const std::string & model_path, const std::string & precision,
    const BatchConfig & batch_config = {1, 1, 1}, const size_t max_workspace_size = (1ULL << 30),
    const BuildConfig & build_config = BuildConfig());
  ~MTRBuilder();

  void setup();

  bool isInitialized() const;

private:
  bool loadEngine(const std::string & filepath);
  bool buildEngineFromOnnx(
    const std::string & filepath, const std::string & output_engine_filepath);

  Logger logger_;
  TrtUniquePtr<nvinfer1::IRuntime> runtime_;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
  TrtUniquePtr<nvinfer1::IExecutionContext> context_;

  fs::path model_filepath_;
  std::string precision_;
  BatchConfig batch_config_;
  size_t max_workspace_size_;
  std::unique_ptr<const BuildConfig> build_config_;

  bool is_initialized_{false};
};  // class MTRBuilder
}  // namespace mtr
#endif  // MTR__BUILDER_HPP_