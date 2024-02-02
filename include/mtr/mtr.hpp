#ifndef MTR__MTR_HPP_
#define MTR__MTR_HPP_

#include "attention/trt_attn_value_computation.hpp"
#include "attention/trt_attn_weight_computation.hpp"
#include "knn/trt_knn_batch.hpp"
#include "knn/trt_knn_batch_mlogk_kernel.hpp"
#include "mtr/builder.hpp"

#include <string>
#include <vector>

namespace mtr
{
class TrtMTR
{
public:
  TrtMTR(
    const std::string & model_path, const std::string & prediction,
    const std::vector<std::string> target_labels =
      {"TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"},
    const BatchConfig & batch_config = {1, 1, 1}, const size_t max_workspace_size = (1ULL << 30),
    const BuildConfig & build_config = BuildConfig());

  bool doInference();

private:
  bool preProcess();
  bool postProcess();

  std::unique_ptr<MTRBuilder> builder_;

  // model parameters
  std::vector<std::string> target_labels_;

  // inputs
  float * in_trajectory_ = nullptr;
  bool * in_trajectory_mask_ = nullptr;
  float * in_polyline_ = nullptr;
  bool * in_polyline_mask_ = nullptr;
  float * in_polyline_center_ = nullptr;
  float * in_last_pos_ = nullptr;
  int * in_track_index_ = nullptr;
  int * in_label_index_ = nullptr;
  // outputs
  float * out_score_ = nullptr;
  float * out_trajectory_ = nullptr;
};  // class TrtMTR
}  // namespace mtr
#endif  // TRT_MTR__NETWORK_HPP_