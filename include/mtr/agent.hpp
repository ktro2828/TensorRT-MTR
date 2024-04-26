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

#ifndef MTR__AGENT_HPP_
#define MTR__AGENT_HPP_

#include <array>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace mtr
{
constexpr size_t AgentStateDim = 12;

enum AgentLabel { VEHICLE = 0, PEDESTRIAN = 1, CYCLIST = 2 };

/**
 * @brief A class to represent a single state of an agent.
 */
struct AgentState
{
  /**
   * @brief Construct a new instance filling all elements by `0.0f`.
   */
  AgentState() : data_({0.0f}) {}

  /**
   * @brief Construct a new instance with specified values.
   *
   * @param x X position.
   * @param y Y position.
   * @param z Z position.
   * @param length Length of the bbox.
   * @param width Width of the bbox.
   * @param height Height of the bbox.
   * @param yaw Heading yaw angle [rad].
   * @param vx Velocity heading x direction in object's coordinates system.
   * @param vy Velocity heading y direction in object's coordinates system.
   * @param ax Acceleration heading x direction in object's coordinates system.
   * @param ay Acceleration heading y direction in object's coordinates system.
   * @param is_valid `1.0f` if valid, otherwise `0.0f`.
   */
  AgentState(
    const float x, const float y, const float z, const float length, const float width,
    const float height, const float yaw, const float vx, const float vy, const float ax,
    const float ay, const float is_valid)
  : data_({x, y, z, length, width, height, yaw, vx, vy, ax, ay, is_valid})
  {
  }

  static const size_t dim = AgentStateDim;

  /**
   * @brief Construct a new instance filling all elements by `0.0f`.
   *
   * @return AgentState
   */
  static AgentState empty() noexcept { return AgentState(); }

  /**
   * @brief Return the address pointer of data array.
   *
   * @return float*
   */
  const float * data_ptr() const noexcept { return data_.data(); }

private:
  std::array<float, dim> data_;
};

/**
 * @brief A class to represent the state history of an agent.
 */
struct AgentHistory
{
  /**
   * @brief Construct a new Agent History filling the latest state by input state.
   *
   * @param state Object current state.
   * @param object_id Object ID.
   * @param current_time Current timestamp.
   * @param max_time_length History length.
   */
  AgentHistory(
    const AgentState & state, const std::string & object_id, const float current_time,
    const size_t max_time_length)
  : data_((max_time_length - 1) * num_state_dim_),
    object_id_(object_id),
    latest_time_(current_time),
    max_time_length_(max_time_length)
  {
    const auto s_ptr = state.data_ptr();
    for (size_t d = 0; d < num_state_dim_; ++d) {
      data_.push_back(*(s_ptr + d));
    }
  }

  /**
   * @brief Construct a new Agent History filling all elements by zero.
   *
   * @param object_id Object ID.
   * @param max_time_length History time length.
   */
  AgentHistory(const std::string & object_id, const size_t max_time_length)
  : data_(max_time_length * num_state_dim_),
    object_id_(object_id),
    latest_time_(-std::numeric_limits<float>::max()),
    max_time_length_(max_time_length)
  {
  }

  // Returns the object ID.
  const std::string & object_id() const { return object_id_; }

  // Return the history length.
  size_t length() const { return max_time_length_; }

  /**
   * @brief Update history with input state and latest time.
   *
   * @param current_time The current timestamp.
   * @param state The current agent state.
   */
  void update(const float current_time, const AgentState & state) noexcept
  {
    // remove the state at the oldest timestamp
    data_.erase(data_.begin(), data_.begin() + num_state_dim_);

    const auto s = state.data_ptr();
    for (size_t d = 0; d < num_state_dim_; ++d) {
      data_.push_back(*(s + d));
    }
    latest_time_ = current_time;
  }

  // Update history with all-zeros state, but latest time is not updated.
  void update_empty() noexcept
  {
    // remove the state at the oldest timestamp
    data_.erase(data_.begin(), data_.begin() + num_state_dim_);

    const auto s = AgentState::empty().data_ptr();
    for (size_t d = 0; d < num_state_dim_; ++d) {
      data_.push_back(*(s + d));
    }
  }

  // Return the address pointer of data array.
  const float * data_ptr() const noexcept { return data_.data(); }

  /**
   * @brief Check whether the latest valid state is too old or not.
   *
   * @param current_time Current timestamp.
   * @param threshold Time difference threshold value.
   * @return true If the difference is greater than threshold.
   * @return false Otherwise
   */
  bool is_ancient(const float current_time, const float threshold) const
  {
    /* TODO: Raise error if the current time is smaller than latest */
    return current_time - latest_time_ >= threshold;
  }

  // Return true if the latest state is valid.
  bool is_valid_latest() const { return data_.at(num_state_dim_ * max_time_length_ - 1) == 1.0f; }

private:
  const size_t num_state_dim_{AgentStateDim};
  std::vector<float> data_;
  const std::string object_id_;
  float latest_time_;
  const size_t max_time_length_;
};

/**
 * @brief A class containing whole state histories of all agent.
 */
struct AgentData
{
  /**
   * @brief Construct a new instance.
   *
   * @param histories An array of histories for each object.
   * @param ego_index An index of ego.
   * @param target_index Indices of target agents.
   * @param label_index An array of label indices for each object.
   * @param timestamps An array of timestamps.
   */
  AgentData(
    const std::vector<AgentHistory> & histories, const int ego_index,
    const std::vector<int> & target_indices, const std::vector<int> & label_indices,
    const std::vector<float> & timestamps)
  : num_target_(target_indices.size()),
    num_agent_(histories.size()),
    num_timestamp_(timestamps.size()),
    ego_index_(ego_index),
    target_indices_(target_indices),
    label_indices_(label_indices),
    timestamps_(timestamps)
  {
    data_.reserve(num_agent_ * num_timestamp_ * num_state_dim_);
    for (auto & history : histories) {
      const auto data_ptr = history.data_ptr();
      for (size_t t = 0; t < num_timestamp_; ++t) {
        for (size_t d = 0; d < num_state_dim_; ++d) {
          data_.push_back(*(data_ptr + t * num_state_dim_ + d));
        }
      }
    }

    target_data_.reserve(num_target_ * num_state_dim_);
    target_label_indices_.reserve(num_target_);
    for (const auto & idx : target_indices_) {
      target_label_indices_.emplace_back(label_indices_.at(idx));
      const auto target_ptr = histories.at(idx).data_ptr();
      for (size_t d = 0; d < num_state_dim_; ++d) {
        target_data_.push_back(*(target_ptr + (num_timestamp_ - 1) * num_state_dim_ + d));
      }
    }

    ego_data_.reserve(num_timestamp_ * num_state_dim_);
    const auto ego_data_ptr = histories.at(ego_index_).data_ptr();
    for (size_t t = 0; t < num_timestamp_; ++t) {
      for (size_t d = 0; d < num_state_dim_; ++d) {
        ego_data_.push_back(*(ego_data_ptr + t * num_state_dim_ + d));
      }
    }
  }

  // Return the number of target agents.
  size_t num_target() const { return num_target_; }

  // Return the number of agents.
  size_t num_agent() const { return num_agent_; }

  // Return the number of timestamps.
  size_t num_timestamp() const { return num_timestamp_; }

  // Return the number of agent state dimensions.
  size_t num_state_dim() const { return num_state_dim_; }

  // Return the number of classes.
  size_t num_class() const { return num_class_; }

  size_t num_attr() const { return num_timestamp_ + num_state_dim_ + num_class_ + 3; }

  // Return the data shape which is meant to `(N, T, D)`.
  std::tuple<size_t, size_t, size_t> shape() const
  {
    return {num_agent_, num_timestamp_, num_state_dim_};
  }

  // Return the number of elements in data which is meant to `N*T*D`.
  size_t size() const { return num_agent_ * num_timestamp_ * num_state_dim_; }

  // Return the number of elements of MTR input (B * N * T * A).
  size_t input_size() const { return num_target_ * num_agent_ * num_timestamp_ * num_attr(); }

  // Return the index number of ego.
  int ego_index() const { return ego_index_; }

  // Return the target indices.
  const std::vector<int> & target_indices() const { return target_indices_; }

  // Return the label indices.
  const std::vector<int> & label_indices() const { return label_indices_; }

  // Return the label indices of targets.
  const std::vector<int> & target_label_indices() const { return target_label_indices_; }

  // Return the timestamps.
  const std::vector<float> & timestamps() const { return timestamps_; }

  // Return the address pointer of data array.
  const float * data_ptr() const noexcept { return data_.data(); }

  /**
   * @brief Return the address pointer of data array for target agents.
   *
   * @return float* The pointer of data array for target agents.
   */
  const float * target_data_ptr() const noexcept { return target_data_.data(); }

  /**
   * @brief Return the address pointer of data array for ego vehicle.
   *
   * @return float* The pointer of data array for ego vehicle.
   */
  const float * ego_data_ptr() const noexcept { return ego_data_.data(); }

private:
  const size_t num_target_;
  const size_t num_agent_;
  const size_t num_timestamp_;
  const size_t num_state_dim_{AgentStateDim};
  const size_t num_class_{3};
  int ego_index_;
  std::vector<int> target_indices_;
  std::vector<int> label_indices_;
  std::vector<float> timestamps_;
  std::vector<int> target_label_indices_;
  std::vector<float> data_;
  std::vector<float> target_data_;
  std::vector<float> ego_data_;
};

std::vector<std::string> getLabelNames(const std::vector<int> & label_index)
{
  std::vector<std::string> label_names;
  label_names.reserve(label_index.size());
  for (const auto & idx : label_index) {
    switch (idx) {
      case 0:
        label_names.emplace_back("VEHICLE");
        break;
      case 1:
        label_names.emplace_back("PEDESTRIAN");
        break;
      case 2:
        label_names.emplace_back("CYCLIST");
        break;
      default:
        std::ostringstream msg;
        msg << "Error invalid label index: " << idx;
        throw std::runtime_error(msg.str());
        break;
    }
  }
  return label_names;
}

}  // namespace mtr
#endif  // MTR__AGENT_HPP_
