#ifndef MTR__POLYLINE_HPP_
#define MTR__POLYLINE_HPP_

#include <array>
#include <cmath>
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
  : data_({x, y, z, dx, dy, dz, label}), x_(x), y_(y), z_(z), label_(label)
  {
  }

  float x() const { return x_; }
  float y() const { return y_; }
  float z() const { return z_; }
  float label() const { return label_; }

  float distance(const LanePoint & other) const
  {
    return std::hypot(x_ - other.x(), y_ - other.y(), z_ - other.z());
  }

  static LanePoint empty() noexcept { return LanePoint(); }

  static const std::size_t Dim = PointStateDim;

  float * data_ptr() noexcept { return data_.data(); }

private:
  std::array<float, Dim> data_;
  float x_, y_, z_, label_;
};

struct PolylineData
{
  PolylineData(
    std::vector<LanePoint> points, const int max_num_polyline, const int max_num_point,
    const float distance_threshold)
  : PolylineNum(0), PointNum(max_num_point), distance_threshold_(distance_threshold)
  {
    std::size_t point_cnt = 0;

    // point_cnt > PointNum at a to a new polyline group
    // distance > threshold -> add to a new polyline group
    for (std::size_t i = 0; i < points.size(); ++i) {
      auto & cur_point = points.at(i);

      if (i == 0) {
        addNewPolyline(cur_point, point_cnt);
        continue;
      }

      if (point_cnt >= PointNum) {
        addNewPolyline(cur_point, point_cnt);
      } else if (const auto & prev_point = points.at(i - 1);
                 cur_point.distance(prev_point) >= distance_threshold_ ||
                 cur_point.label() != prev_point.label()) {
        if (point_cnt < PointNum) {
          addEmptyPoints(point_cnt);
        }
        addNewPolyline(cur_point, point_cnt);
      } else {
        addPoint(cur_point, point_cnt);
      }
    }

    if (PolylineNum < max_num_polyline) {
      addEmptyPolyline(max_num_polyline - PolylineNum);
    }
  }

  std::size_t PolylineNum;
  std::size_t PointNum;
  const std::size_t StateDim = PointStateDim;

  float * data_ptr() noexcept { return data_.data(); }

private:
  void addEmptyPolyline(std::size_t num_polyline)
  {
    for (std::size_t i = 0; i < num_polyline; ++i) {
      std::size_t point_cnt = 0;
      auto empty_point = LanePoint::empty();
      addNewPolyline(empty_point, point_cnt);
      addEmptyPoints(point_cnt);
    }
  }

  void addNewPolyline(LanePoint & point, std::size_t & point_cnt)
  {
    const auto s = point.data_ptr();
    for (std::size_t d = 0; d < StateDim; ++d) {
      data_.push_back(*(s + d));
    }
    ++PolylineNum;
    point_cnt = 1;
  }

  void addEmptyPoints(std::size_t & point_cnt)
  {
    const auto s = LanePoint::empty().data_ptr();
    for (std::size_t n = point_cnt; n < PointNum; ++n) {
      for (std::size_t d = 0; d < StateDim; ++d) {
        data_.push_back(*(s + d));
      }
    }
    point_cnt = PointNum;
  }

  void addPoint(LanePoint & point, std::size_t & point_cnt)
  {
    const auto s = point.data_ptr();
    for (std::size_t d = 0; d < StateDim; ++d) {
      data_.push_back(*(s + d));
    }
    ++point_cnt;
  }

  std::vector<float> data_;
  const float distance_threshold_;
};

}  // namespace mtr
#endif  // MTR__POLYLINE_HPP_