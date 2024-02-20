#ifndef MTR__MTR_HPP_
#define MTR__MTR_HPP_

#include "attention/trt_attn_value_computation.hpp"
#include "attention/trt_attn_weight_computation.hpp"
#include "knn/trt_knn_batch.hpp"
#include "knn/trt_knn_batch_mlogk_kernel.hpp"
#include "mtr/agent.hpp"
#include "mtr/builder.hpp"
#include "mtr/cuda_helper.hpp"
#include "mtr/polyline.hpp"

#include <string>
#include <vector>

namespace mtr
{
struct MtrConfig
{
  MtrConfig(
    const std::vector<std::string> & target_labels = {"VEHICLE", "PEDESTRIAN", "CYCLIST"},
    const size_t num_mode = 6, const size_t num_future = 80, const size_t max_num_polyline = 768)
  : target_labels(target_labels),
    num_mode(num_mode),
    num_future(num_future),
    max_num_polyline(max_num_polyline)
  {
  }
  std::vector<std::string> target_labels;
  size_t num_mode;
  size_t num_future;
  size_t max_num_polyline;
};

class TrtMTR
{
public:
  TrtMTR(
    const std::string & model_path, const std::string & precision,
    const MtrConfig & config = MtrConfig(), const BatchConfig & batch_config = {1, 1, 1},
    const size_t max_workspace_size = (1ULL << 30),
    const BuildConfig & build_config = BuildConfig());

  bool doInference(AgentData & agent_data, PolylineData & polyline_data);

  const MtrConfig & config() const { return config_; }

private:
  void initCudaPtr(AgentData & agent_data, PolylineData & polyline_data);
  bool preProcess(AgentData & agent_data, PolylineData & polyline_data);
  bool postProcess(AgentData & agent_data);

  // model parameters
  MtrConfig config_;

  std::unique_ptr<MTRBuilder> builder_;
  cudaStream_t stream_{nullptr};

  // source data
  cuda::unique_ptr<int[]> d_target_index_{nullptr};
  cuda::unique_ptr<int[]> d_label_index_{nullptr};
  cuda::unique_ptr<float[]> d_timestamps_{nullptr};
  cuda::unique_ptr<float[]> d_trajectory_{nullptr};
  cuda::unique_ptr<float[]> d_target_state_{nullptr};
  cuda::unique_ptr<float[]> d_intention_points_{nullptr};
  cuda::unique_ptr<float[]> d_polyline_{nullptr};
  cuda::unique_ptr<int[]> d_topk_index_{nullptr};

  // preprocessed inputs
  cuda::unique_ptr<float[]> d_in_trajectory_{nullptr};
  cuda::unique_ptr<bool[]> d_in_trajectory_mask_{nullptr};
  cuda::unique_ptr<float[]> d_in_last_pos_{nullptr};
  cuda::unique_ptr<float[]> d_in_polyline_{nullptr};
  cuda::unique_ptr<bool[]> d_in_polyline_mask_{nullptr};
  cuda::unique_ptr<float[]> d_in_polyline_center_{nullptr};

  // outputs
  cuda::unique_ptr<float[]> d_out_score_{nullptr};
  cuda::unique_ptr<float[]> d_out_trajectory_{nullptr};

  // debug
  cuda::EventDebugger event_debugger_;
  std::unique_ptr<float[]> h_debug_in_trajectory_{nullptr};
  std::unique_ptr<bool[]> h_debug_in_trajectory_mask_{nullptr};
  std::unique_ptr<float[]> h_debug_in_last_pos_{nullptr};
  std::unique_ptr<float[]> h_debug_in_polyline_{nullptr};
  std::unique_ptr<bool[]> h_debug_in_polyline_mask_{nullptr};
  std::unique_ptr<float[]> h_debug_in_polyline_center_{nullptr};
  std::unique_ptr<float[]> h_debug_out_score_{nullptr};
  std::unique_ptr<float[]> h_debug_out_trajectory_{nullptr};

  void debugPreprocess(const AgentData & agent_data);
  void debugPostprocess(const AgentData & agent_data);
};  // class TrtMTR
}  // namespace mtr
#endif  // MTR__NETWORK_HPP_