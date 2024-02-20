#include "mtr/polyline.hpp"

#include <cassert>
#include <iostream>
#include <vector>

int main()
{
  constexpr int N = 10;
  constexpr int K = 20;
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

  mtr::PolylineData data(points, K, P, 2.0f);

  std::cout << "=== Polyline data ===\n";
  const float * data_ptr = data.data_ptr();
  for (std::size_t k = 0; k < data.PolylineNum; ++k) {
    std::cout << "Batch " << k << ":\n";
    for (std::size_t p = 0; p < data.PointNum; ++p) {
      std::cout << "  Point " << p << ": ";
      for (std::size_t d = 0; d < data.StateDim; ++d) {
        std::cout << *(data_ptr + k * P * D + p * D + d) << " ";
      }
      std::cout << "\n";
    }
  }
}