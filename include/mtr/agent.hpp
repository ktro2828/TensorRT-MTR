#ifndef MTR__AGENT_HPP_
#define MTR__AGENT_HPP_

#include <array>
#include <string>
#include <vector>

namespace mtr
{
constexpr size_t AgentStateDim = 12;

enum AgentLabel { VEHICLE = 0, PEDESTRIAN = 1, CYCLIST = 2 };

struct AgentState
{
  AgentState() : data_({0.0f}) {}

  AgentState(
    const float x, const float y, const float z, const float length, const float width,
    const float height, const float yaw, const float vx, const float vy, const float ax,
    const float ay, const float is_valid)
  : data_({x, y, z, length, width, height, yaw, vx, vy, ax, ay, is_valid})
  {
  }

  static AgentState empty() noexcept { return AgentState(); }

  static const size_t Dim = AgentStateDim;

  float * data_ptr() noexcept { return data_.data(); }

private:
  std::array<float, Dim> data_;
};

struct AgentHistory
{
  explicit AgentHistory(const std::string & object_id) : object_id_(object_id) {}

  AgentHistory(const std::string & object_id, const size_t max_time_length) : object_id_(object_id)
  {
    data_.reserve(max_time_length * StateDim);
  }

  static const size_t StateDim = AgentStateDim;
  const std::string & object_id() const { return object_id_; }
  size_t length() const { return data_.size() / StateDim; }

  void push_back(const float current_time, AgentState & state) noexcept
  {
    const float * s = state.data_ptr();
    for (size_t d = 0; d < StateDim; ++d) {
      data_.push_back(*(s + d));
    }
    latest_time_ = current_time;
  }
  float * data_ptr() noexcept { return data_.data(); }
  float latest_time() const { return latest_time_; }
  bool has_current_data(const float current_time) const { return latest_time_ == current_time; }
  bool is_valid(const float current_time, const float threshold) const
  {
    return current_time - latest_time_ < threshold;
  }

private:
  std::vector<float> data_;
  std::string object_id_;
  float latest_time_;
};

struct AgentData
{
  AgentData(
    std::vector<AgentHistory> & histories, const int sdc_index,
    const std::vector<int> & target_index, const std::vector<int> & label_index,
    const std::vector<float> & timestamps)
  : TargetNum(target_index.size()),
    AgentNum(histories.size()),
    TimeLength(timestamps.size()),
    sdc_index(sdc_index),
    target_index(target_index),
    label_index(label_index),
    timestamps(timestamps)
  {
    data_.reserve(AgentNum * TimeLength * StateDim);
    for (auto & history : histories) {
      const auto data_ptr = history.data_ptr();
      for (size_t t = 0; t < TimeLength; ++t) {
        for (size_t d = 0; d < StateDim; ++d) {
          data_.push_back(*(data_ptr + t * StateDim + d));
        }
      }
    }
  }

  const size_t TargetNum;
  const size_t AgentNum;
  const size_t TimeLength;
  const size_t StateDim = AgentStateDim;
  const size_t ClassNum = 3;  // TODO

  int sdc_index;
  std::vector<int> target_index;
  std::vector<int> label_index;
  std::vector<float> timestamps;

  float * data_ptr() noexcept { return data_.data(); }

private:
  std::vector<float> data_;
};

}  // namespace mtr
#endif  // MTR__AGENT_HPP_