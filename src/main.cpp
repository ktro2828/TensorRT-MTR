#include "mtr/mtr.hpp"

#include <math.h>

#include <cassert>
#include <string>

static const std::string USAGE = "[USAGE]: ./build/main <PATH_TO_ONNX> [or <PATH_TO_ENGINE>]";

mtr::AgentData load_agent_data()
{
  constexpr int B = 2;
  constexpr int N = 4;
  constexpr int T = 5;
  constexpr int D = 12;
  constexpr int sdc_index = 1;

  std::vector<int> target_index{0, 2};
  std::vector<int> label_index{0, 0, 2, 1};
  std::vector<float> timestamps{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  float trajectory[N][T][D] = {
    {
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, M_PI / 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
      {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, M_PI / 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f},
      {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, M_PI / 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
      {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, M_PI / 2.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, M_PI / 2.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
    },
    {
      {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, M_PI / 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 0.0f},
      {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, M_PI / 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 0.0f},
      {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, M_PI / 2.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, M_PI / 2.0f, 5.0f, 5.0f, 5.0f, 5.0f, 0.0f},
      {6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, M_PI / 2.0f, 6.0f, 6.0f, 6.0f, 6.0f, 1.0f},
    },
    {
      {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, M_PI / 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
      {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, M_PI / 2.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, M_PI / 2.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
      {6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, M_PI / 2.0f, 6.0f, 6.0f, 6.0f, 6.0f, 1.0f},
      {7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f, M_PI / 2.0f, 7.0f, 7.0f, 7.0f, 7.0f, 1.0f},
    },
    {
      {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, M_PI / 2.0f, 4.0f, 4.0f, 4.0f, 4.0f, 0.0f},
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, M_PI / 2.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
      {6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, M_PI / 2.0f, 6.0f, 6.0f, 6.0f, 6.0f, 0.0f},
      {7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f, M_PI / 2.0f, 7.0f, 7.0f, 7.0f, 7.0f, 0.0f},
      {8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, M_PI / 2.0f, 8.0f, 8.0f, 8.0f, 8.0f, 1.0f},
    },
  };

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

int main(int argc, char ** argv)
{
  assert((USAGE, argc == 2));
  auto model_path = std::string(argv[1]);

  auto model = std::make_unique<mtr::TrtMTR>(model_path, "FP32");

  auto agent_data = load_agent_data();

  if (!model->doInference(agent_data)) {
    std::cerr << "===== [FAIL]: Fail to inference!! =====" << std::endl;
  } else {
    std::cout << "===== [SUCCESS] Success to inference!! =====" << std::endl;
  }
}