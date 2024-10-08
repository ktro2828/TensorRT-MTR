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

#ifndef MTR__TRAJECTORY_HPP_
#define MTR__TRAJECTORY_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace mtr
{

constexpr size_t PredictedStateDim = 7;

/**
 * @brief A class to represent a predicted state.
 */
struct PredictedState
{
  explicit PredictedState(const std::array<float, PredictedStateDim> & state)
  : x_(state.at(0)),
    y_(state.at(1)),
    dx_(state.at(2)),
    dy_(state.at(3)),
    yaw_(state.at(4)),
    vx_(state.at(5)),
    vy_(state.at(6))
  {
  }

  PredictedState(
    const float x, const float y, const float dx, const float dy, const float yaw, const float vx,
    const float vy)
  : x_(x), y_(y), dx_(dx), dy_(dy), yaw_(yaw), vx_(vx), vy_(vy)
  {
  }

  static const size_t dim = PredictedStateDim;

  float x() const { return x_; }
  float y() const { return y_; }
  float dx() const { return dx_; }
  float dy() const { return dy_; }
  float yaw() const { return yaw_; }
  float vx() const { return vx_; }
  float vy() const { return vy_; }

private:
  float x_, y_, dx_, dy_, yaw_, vx_, vy_;
};  // struct PredictedState

/**
 * @brief A class to represent waypoints for a single motion mode.
 */
struct PredictedMode
{
  PredictedMode(const float score, const std::vector<float> & waypoints, const size_t num_future)
  : score_(score), num_future_(num_future)
  {
    for (size_t t = 0; t < num_future_; ++t) {
      const auto start_itr = waypoints.cbegin() + t * num_state_dim_;
      std::array<float, PredictedStateDim> state;
      std::copy_n(start_itr, PredictedStateDim, state.begin());
      waypoints_.emplace_back(state);
    }
  }

  float score() const { return score_; }

  size_t num_future() const { return num_future_; }

  size_t num_state_dim() const { return num_state_dim_; }

  //  Get the waypoints at a single mode.
  const std::vector<PredictedState> & get_waypoints() const { return waypoints_; }

private:
  float score_;
  size_t num_future_;
  size_t num_state_dim_{PredictedStateDim};
  std::vector<PredictedState> waypoints_;
};  // struct PredictedMode

/**
 * @brief A class to represent predicted trajectory for a single target agent.
 */
struct PredictedTrajectory
{
  PredictedTrajectory(
    const std::vector<float> & scores, const std::vector<float> & modes, const size_t num_mode,
    const size_t num_future)
  : num_mode_(num_mode), num_future_(num_future)
  {
    for (size_t m = 0; m < num_mode_; ++m) {
      const auto score = scores.at(m);
      const auto wp_itr = modes.cbegin() + m * num_future_ * num_state_dim_;
      std::vector<float> waypoints(wp_itr, wp_itr + num_future_ * num_state_dim_);
      modes_.emplace_back(score, waypoints, num_future_);
    }

    // sort by score
    sortByScore();
  }

  // Return the number of modes.
  size_t num_mode() const { return num_mode_; }

  // Return the number of future timestamps.
  size_t num_future() const { return num_future_; }

  // Return the number of state dimensions.
  size_t num_state_dim() const { return num_state_dim_; }

  // Return predicted modes. Modes are sorted in descending order based on their scores.
  const std::vector<PredictedMode> & get_modes() const { return modes_; }

private:
  /**
   * @brief Sort modes in descending order based on their scores.
   */
  void sortByScore()
  {
    std::sort(modes_.begin(), modes_.end(), [](const auto & mode1, const auto & mode2) {
      return mode1.score() > mode2.score();
    });
  }

  const size_t num_mode_;
  const size_t num_future_;
  const size_t num_state_dim_{PredictedStateDim};
  std::vector<PredictedMode> modes_;
};  // struct PredictedTrajectory
}  // namespace mtr
#endif  // MTR__TRAJECTORY_HPP_
