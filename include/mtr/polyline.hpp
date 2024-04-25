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

#ifndef MTR__POLYLINE_HPP_
#define MTR__POLYLINE_HPP_

#include <array>
#include <cmath>
#include <tuple>
#include <vector>

namespace mtr
{
constexpr std::size_t PointStateDim = 7;

enum PolylineLabel { LANE = 0, ROAD_LINE = 1, ROAD_EDGE = 2, CROSSWALK = 3 };

struct LanePoint
{
  /**
   * @brief Construct a new instance filling all elements by `0.0f`.
   */
  LanePoint() : data_({0.0f}) {}

  /**
   * @brief Construct a new instance with specified values.
   *
   * @param x X position.
   * @param y Y position.
   * @param z Z position.
   * @param dx Normalized delta x.
   * @param dy Normalized delta y.
   * @param dz Normalized delta z.
   * @param label Label.
   */
  LanePoint(
    const float x, const float y, const float z, const float dx, const float dy, const float dz,
    const float label)
  : data_({x, y, z, dx, dy, dz, label}), x_(x), y_(y), z_(z), label_(label)
  {
  }

  static const std::size_t dim{PointStateDim};

  /**
   * @brief Return the x position of the point.
   *
   * @return float The x position.
   */
  float x() const { return x_; }

  /**
   * @brief Return the y position of the point.
   *
   * @return float The y position.
   */
  float y() const { return y_; }

  /**
   * @brief Return the z position of the point.
   *
   * @return float The z position.
   */
  float z() const { return z_; }

  /**
   * @brief Return the label of the point.
   *
   * @return float The label.
   */
  float label() const { return label_; }

  /**
   * @brief Return the distance between myself and another one.
   *
   * @param other Another point.
   * @return float Distance between myself and another one.
   */
  float distance(const LanePoint & other) const
  {
    return std::hypot(x_ - other.x(), y_ - other.y(), z_ - other.z());
  }

  /**
   * @brief Construct a new instance filling all elements by `0.0f`.
   *
   * @return LanePoint
   */
  static LanePoint empty() noexcept { return LanePoint(); }

  /**
   * @brief Return the address pointer of data array.
   *
   * @return float*
   */
  const float * data_ptr() const noexcept { return data_.data(); }

private:
  std::array<float, dim> data_;
  float x_, y_, z_, label_;
};

struct PolylineData
{
  /**
   * @brief Construct a new PolylineData instance.
   *
   * @param points Source points vector.
   * @param min_num_polyline The minimum number of polylines should be generated. If the number of
   * polylines, resulting in separating input points, is less than this value, empty polylines will
   * be added.
   * @param max_num_point The maximum number of points that each polyline can include. If the
   * polyline contains fewer points than this value, empty points will be added.
   * @param distance_threshold The distance threshold to separate polylines.
   */
  PolylineData(
    const std::vector<LanePoint> & points, const int min_num_polyline, const int max_num_point,
    const float distance_threshold)
  : num_polyline_(0), num_point_(max_num_point), distance_threshold_(distance_threshold)
  {
    std::size_t point_cnt = 0;

    // point_cnt > num_point_ at a to a new polyline group
    // distance > threshold -> add to a new polyline group
    for (std::size_t i = 0; i < points.size(); ++i) {
      auto & cur_point = points.at(i);

      if (i == 0) {
        addNewPolyline(cur_point, point_cnt);
        continue;
      }

      if (point_cnt >= num_point_) {
        addNewPolyline(cur_point, point_cnt);
      } else if (const auto & prev_point = points.at(i - 1);
                 cur_point.distance(prev_point) >= distance_threshold_ ||
                 cur_point.label() != prev_point.label()) {
        if (point_cnt < num_point_) {
          addEmptyPoints(point_cnt);
        }
        addNewPolyline(cur_point, point_cnt);
      } else {
        addPoint(cur_point, point_cnt);
      }
    }
    addEmptyPoints(point_cnt);

    if (num_polyline_ < min_num_polyline) {
      addEmptyPolyline(min_num_polyline - num_polyline_);
    }
  }

  // Return the total number of polylines.
  size_t num_polyline() const { return num_polyline_; }

  // Return the number of points contained in each polyline.
  size_t num_point() const { return num_point_; }

  // Return the number of point dimensions.
  size_t num_state_dim() const { return num_state_dim_; }

  // Return the data shape which is meant to `(L, P, D)`.
  std::tuple<size_t, size_t, size_t> shape() const
  {
    return {num_polyline_, num_point_, num_state_dim_};
  }

  // Return the total number of elements which is meant to `L*P*D`.
  size_t size() const { return num_polyline_ * num_point_ * num_state_dim_; }

  // Return the input attribute size which is meant to `D+2`.
  size_t input_attribute_size() const { return num_state_dim_ + 2; }

  // Return the address pointer of data array.
  const float * data_ptr() const noexcept { return data_.data(); }

private:
  /**
   * @brief Add a new polyline group filled by empty points. This member function increments
   * `num_polyline_` by `num_polyline` internally.
   *
   * @param num_polyline The number of polylines to add.
   */
  void addEmptyPolyline(std::size_t num_polyline)
  {
    for (std::size_t i = 0; i < num_polyline; ++i) {
      std::size_t point_cnt = 0;
      auto empty_point = LanePoint::empty();
      addNewPolyline(empty_point, point_cnt);
      addEmptyPoints(point_cnt);
    }
  }

  /**
   * @brief Add a new polyline group with the specified point. This member function increments
   * `num_polyline_` by `1` internally.
   *
   * @param point LanePoint instance.
   * @param point_cnt The current count of points, which will be reset to `1`.
   */
  void addNewPolyline(const LanePoint & point, std::size_t & point_cnt)
  {
    const auto s = point.data_ptr();
    for (std::size_t d = 0; d < num_state_dim_; ++d) {
      data_.push_back(*(s + d));
    }
    ++num_polyline_;
    point_cnt = 1;
  }

  /**
   * @brief Add `(num_point_ - point_cnt)` empty points filled by `0.0`.
   *
   * @param point_cnt The number of current count of points, which will be reset to `num_point_`.
   */
  void addEmptyPoints(std::size_t & point_cnt)
  {
    const auto s = LanePoint::empty().data_ptr();
    for (std::size_t n = point_cnt; n < num_point_; ++n) {
      for (std::size_t d = 0; d < num_state_dim_; ++d) {
        data_.push_back(*(s + d));
      }
    }
    point_cnt = num_point_;
  }

  /**
   * @brief Add the specified point and increment `point_cnt` by `1`.
   *
   * @param point
   * @param point_cnt
   */
  void addPoint(const LanePoint & point, std::size_t & point_cnt)
  {
    const auto s = point.data_ptr();
    for (std::size_t d = 0; d < num_state_dim_; ++d) {
      data_.push_back(*(s + d));
    }
    ++point_cnt;
  }

  size_t num_polyline_;
  const std::size_t num_point_;
  const std::size_t num_state_dim_{PointStateDim};
  std::vector<float> data_;
  const float distance_threshold_;
};

}  // namespace mtr
#endif  // MTR__POLYLINE_HPP_
