#include "mtr/intention_point.hpp"

#include <iostream>
#include <string>
#include <vector>

int main()
{
  const std::string filename = "./data/waymo64.csv";
  constexpr size_t num_cluster = 64;
  auto intention_point = mtr::IntentionPoint(filename, num_cluster);

  std::vector<std::string> label_names = {"VEHICLE", "PEDESTRIAN", "CYCLIST", "VEHICLE"};
  std::vector<float> points = intention_point.get_points(label_names);

  for (size_t i = 0; i < label_names.size(); ++i) {
    std::cout << "Label " << label_names.at(i) << ":\n";
    for (size_t k = 0; k < num_cluster; ++k) {
      std::cout << "  K " << k << ": ";
      for (size_t d = 0; d < mtr::IntentionPoint::StateDim; ++d) {
        std::cout << points.at((i * num_cluster + k) * mtr::IntentionPoint::StateDim + d) << " ";
      }
      std::cout << "\n";
    }
  }
}