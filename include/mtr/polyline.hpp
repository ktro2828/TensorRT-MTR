#ifndef MTR__POLYLINE_HPP_
#define MTR__POLYLINE_HPP_

#include <array>
#include <vector>

namespace mtr
{
constexpr std::size_t PointStateDim = 7;

enum PolylineLabel { LANE = 0, ROAD_LINE = 1, ROAD_EDGE = 2, CROSSWALK = 3 };

struct LanePoint
{
  LanePoint() : data_({0.0f}) {}

  LanePoint(
    const float x, const float y, const float z, const float dx, const float dy, const float dz,
    const float label)
  : data_({x, y, z, dx, dy, dz, label})
  {
  }

  static LanePoint empty() noexcept { return LanePoint(); }

  static const std::size_t Dim = PointStateDim;

  float * data_ptr() noexcept { return data_.data(); }

private:
  std::array<float, Dim> data_;
};

struct LanePolyline
{
  float * data_ptr() noexcept { return data_.data(); }

private:
  std::vector<float> data_;
};

struct PolylineData
{
  PolylineData(std::vector<LanePolyline> & polylines) {}

  float * data_ptr() noexcept { return data_.data(); }

private:
  std::vector<float> data_;
};
}  // namespace mtr
#endif  // MTR__POLYLINE_HPP_