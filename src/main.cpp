#include "mtr/debugger.hpp"
#include "mtr/mtr.hpp"

#include <math.h>

#include <cassert>
#include <string>

static const std::string USAGE =
  "[USAGE]: ./build/main <PATH_TO_ONNX>(or <PATH_TO_ENGINE>) [<NUM_REPEAT=1>]";

mtr::AgentData load_agent_data()
{
  // NOTE: current expected input size is (3, 55, 11, 12)
  constexpr int B = 3;
  constexpr int N = 55;
  constexpr int T = 11;
  constexpr int D = 12;
  constexpr int sdc_index = 1;

  std::vector<int> target_index{0, 1, 2};
  std::vector<float> timestamps{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};

  std::vector<int> label_index;
  float trajectory[N][T][D];
  for (int n = 0; n < N; ++n) {
    label_index.emplace_back(0);
    for (int t = 0; t < T; ++t) {
      for (int d = 0; d < D; ++d) {
        trajectory[n][t][d] = static_cast<float>(n * t + d);
      }
    }
  }

  std::vector<mtr::AgentHistory> histories;
  histories.reserve(N);
  for (size_t n = 0; n < N; ++n) {
    mtr::AgentHistory history("foo", T);
    for (size_t t = 0; t < T; ++t) {
      const size_t idx = n * T * D + t * D;
      auto x = trajectory[n][t][0];
      auto y = trajectory[n][t][1];
      auto z = trajectory[n][t][2];
      auto length = trajectory[n][t][3];
      auto width = trajectory[n][t][4];
      auto height = trajectory[n][t][5];
      auto yaw = trajectory[n][t][6];
      auto vx = trajectory[n][t][7];
      auto vy = trajectory[n][t][8];
      auto ax = trajectory[n][t][9];
      auto ay = trajectory[n][t][10];
      auto is_valid = trajectory[n][t][11];
      mtr::AgentState state(x, y, z, length, width, height, yaw, vx, vy, ax, ay, is_valid);
      history.update(static_cast<float>(t), state);
    }
    histories.emplace_back(history);
  }
  return mtr::AgentData(histories, sdc_index, target_index, label_index, timestamps);
}

mtr::PolylineData load_polyline_data(const int K)
{
  constexpr int N = 10;
  constexpr int P = 5;
  constexpr int D = 7;

  float src_points[N][D] = {
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f},
    {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f}, {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
    {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
    {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}, {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
    {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f}, {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
  };

  std::vector<mtr::LanePoint> points;
  points.reserve(N);
  for (std::size_t n = 0; n < N; ++n) {
    const std::size_t idx = n * D;
    auto x = src_points[n][0];
    auto y = src_points[n][1];
    auto z = src_points[n][2];
    auto dx = src_points[n][3];
    auto dy = src_points[n][4];
    auto dz = src_points[n][5];
    auto label = src_points[n][6];
    mtr::LanePoint point(x, y, z, dx, dy, dz, label);
    points.emplace_back(point);
  }

  return mtr::PolylineData(points, K, P, 2.0f);
}

int main(int argc, char ** argv)
{
  assert((USAGE, argc == 2 || argc == 3));
  auto model_path = std::string(argv[1]);
  const int num_repeat = argc == 3 ? atoi(argv[2]) : 1;
  assert(("The number of repeats must be integer > 0", num_repeat > 0));

  mtr::Debugger debugger;
  auto model = std::make_unique<mtr::TrtMTR>(model_path, "FP32");

  debugger.createEvent();
  auto agent_data = load_agent_data();
  auto polyline_data = load_polyline_data(model->config().max_num_polyline);
  debugger.printElapsedTime("Data loading time: ");

  for (int i = 0; i < num_repeat; ++i) {
    debugger.createEvent();
    if (!model->doInference(agent_data, polyline_data)) {
      std::cerr << "===== [FAIL]: Fail to inference!! =====" << std::endl;
    } else {
      std::cout << "===== [SUCCESS] Success to inference!! =====" << std::endl;
    }
    debugger.printElapsedTime("Inference time: ");
  }
}