#ifndef MTR__INTENTION_POINT_HPP_
#define MTR__INTENTION_POINT_HPP_

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace mtr
{

constexpr size_t IntentionPointDim = 2;

struct IntentionPoint
{
public:
  /**
   * @brief Construct a new Intention.
   *
   * @param data_map Map of intention points hashed by label names.
   * @param num_cluster The number of clusters.
   */
  IntentionPoint(
    const std::unordered_map<std::string, std::vector<float>> data_map, const size_t num_cluster)
  : data_map_(data_map), num_cluster_(num_cluster)
  {
  }

  static const size_t StateDim = IntentionPointDim;

  /**
   * @brief Construct a new instance from csv file.
   *
   * @param filename Path of csv file (.csv).
   * @param num_cluster The number of clusters.
   */
  static IntentionPoint fromCsv(const std::string & filename, const size_t num_cluster)
  {
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::ostringstream err_msg;
      err_msg << "Error opening file: " << filename << ". Please check if the file exists.";
      throw std::runtime_error(err_msg.str());
    }

    std::vector<std::tuple<float, float, std::string>> buffer;
    std::string line;
    while (std::getline(file, line)) {
      std::istringstream ss(line);
      float x, y;
      std::string label;

      ss >> x;
      ss.ignore();
      ss >> y;
      ss.ignore();
      std::getline(ss, label, ',');

      buffer.emplace_back(x, y, label);
    }
    file.close();

    std::unordered_map<std::string, std::vector<float>> data_map;
    for (const auto & [x, y, label] : buffer) {
      data_map[label].emplace_back(x);
      data_map[label].emplace_back(y);
    }

    for (const auto & [key, values] : data_map) {
      assert(
        ("The number of clusters is not same with the specified value",
         values.size() == StateDim * num_cluster));
    }

    return IntentionPoint(data_map, num_cluster);
  }

  /**
   * @brief Load intention points for specified label names.
   *
   * @param label_names Array of label names for all agents, in shape [N].
   * @return std::vector<float> Array of all points in shape, [N * num_cluster * 2].
   */
  std::vector<float> get_points(std::vector<std::string> & label_names)
  {
    std::vector<float> points;
    points.reserve(label_names.size() * num_cluster_ * StateDim);
    for (const auto & name : label_names) {
      const auto & label_points = data_map_.at(name);
      for (const auto & p : label_points) {
        points.emplace_back(p);
      }
    }
    return points;
  }

  /**
   * @brief Return the number of clusters contained in intention points.
   *
   * @return size_t
   */
  size_t num_cluster() const { return num_cluster_; }

private:
  std::unordered_map<std::string, std::vector<float>> data_map_;
  size_t num_cluster_;
};
}  // namespace mtr
#endif  // MTR__INTENTION_POINT_HPP_