#include "mtr/agent.hpp"

#include <array>
#include <iostream>
#include <string>
#include <vector>

int main()
{
  constexpr size_t N = 5;
  constexpr size_t T = 10;
  std::vector<mtr::AgentHistory> histories;
  int sdc_index = 1;
  std::vector<int> target_index = {0, 3};
  std::vector<int> label_index = {0, 0, 2, 1, 2};
  std::vector<float> timestamps;

  for (size_t n = 0; n < N; ++n) {
    mtr::AgentHistory history("foo", T);
    for (size_t t = 0; t < T; ++t) {
      if (t % 2 != 0) {
        const float v = static_cast<float>(t * (n + 1));
        mtr::AgentState state(v, v, v, v, v, v, v, v, v, v, v, v);
        history.push_back(static_cast<float>(t), state);
      } else {
        auto state = mtr::AgentState::empty();
        history.push_back(static_cast<float>(t), state);
      }
      if (n == 0) {
        timestamps.push_back(static_cast<float>(t));
      }
    }
    histories.emplace_back(history);
  }

  mtr::AgentData data(histories, sdc_index, target_index, label_index, timestamps);

  const float * data_ptr = data.data_ptr();
  const size_t D = mtr::AgentStateDim;
  for (size_t n = 0; n < N; ++n) {
    std::cout << "Agent: " << n << "\n";
    for (size_t t = 0; t < T; ++t) {
      for (size_t d = 0; d < D; ++d) {
        std::cout << *(data_ptr + n * T * D + t * D + d) << " ";
      }
      std::cout << "\n";
    }
  }
}
