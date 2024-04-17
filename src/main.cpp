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

#include "mtr/agent.hpp"
#include "mtr/debugger.hpp"
#include "mtr/mtr.hpp"
#include "mtr/polyline.hpp"
#include "mtr/trajectory.hpp"

#include <cassert>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static const std::string USAGE =
  "[USAGE]: ./build/main <PATH_TO_ONNX>(or <PATH_TO_ENGINE>) [<NUM_REPEAT=1>]";

/**
 * @brief Load timestamps from dat file.
 *
 * @param filepath
 * @return std::vector<float>
 */
std::vector<float> loadTimestamp(const std::string & filepath)
{
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::ostringstream err_msg;
    err_msg << "Failed to open timestamp file: " << filepath;
    throw std::runtime_error(err_msg.str());
  }

  std::vector<float> timestamps;
  while (!file.eof()) {
    float t;
    file >> t;
    timestamps.emplace_back(t);
  }
  file.close();
  return timestamps;
}

/**
 * @brief Load `mtr::AgentData` from dat files.
 *
 * @return mtr::AgentData
 */
mtr::AgentData loadAgentData()
{
  const std::vector<float> timestamps = loadTimestamp("./data/Timestamp.dat");
  const auto max_time_length = timestamps.size();

  // [agent_id, filepath]: (B, N) = (2, 7)
  const std::map<std::string, std::string> filepaths = {
    {"Target0", "./data/TargetVehicle0.dat"},
    {"Target1", "./data/TargetVehicle1.dat"},
    {"Agent0", "./data/AgentVehicle0.dat"},
    {"Agent1", "./data/AgentVehicle1.dat"},
    {"Agent2", "./data/AgentVehicle2.dat"},
    {"Agent3", "./data/AgentVehicle3.dat"},
    {"EGO", "./data/Ego.dat"}};

  const std::vector<int> target_index = {0, 1};
  const std::vector<int> label_index = {0, 0, 0, 0, 0, 0, 0};
  constexpr int sdc_index = 6;

  std::vector<mtr::AgentHistory> histories;
  for (const auto & [agent_id, filepath] : filepaths) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
      std::ostringstream err_msg;
      err_msg << "Failed to open agent file: " << filepath;
      throw std::runtime_error(err_msg.str());
    }

    mtr::AgentHistory history(agent_id, max_time_length);
    size_t cnt = 0;
    while (!file.eof()) {
      auto current_time = timestamps.at(cnt);

      float x, y, z, length, width, height, yaw, vx, vy, ax, ay, is_valid;
      file >> x >> y >> z >> length >> width >> height >> yaw >> vx >> vy >> ax >> ay >> is_valid;
      mtr::AgentState state(x, y, z, length, width, height, yaw, vx, vy, ax, ay, is_valid);
      history.update(current_time, state);
      ++cnt;
    }
    file.close();
    histories.emplace_back(history);
  }
  return mtr::AgentData(histories, sdc_index, target_index, label_index, timestamps);
}

/**
 * @brief Load `mtr::PolylineData` from dat file.
 *
 * @return mtr::PolylineData
 */
mtr::PolylineData loadPolylineData(const size_t K, const size_t P, const float threshold)
{
  const std::string filepath = "./data/Polyline.dat";

  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::ostringstream err_msg;
    err_msg << "Failed to open polyline file: " << filepath;
    throw std::runtime_error(err_msg.str());
  }

  std::vector<mtr::LanePoint> points;
  while (!file.eof()) {
    float x, y, z, dx, dy, dz, label;
    file >> x >> y >> z >> dx >> dy >> dz >> label;
    points.emplace_back(x, y, z, dx, dy, dz, label);
  }
  file.close();
  return mtr::PolylineData(points, K, P, threshold);
}

int main(int argc, char ** argv)
{
  assert((USAGE, argc == 2 || argc == 3));
  auto model_path = std::string(argv[1]);
  const int num_repeat = argc == 3 ? atoi(argv[2]) : 1;
  assert(("The number of repeats must be integer > 0", num_repeat > 0));

  mtr::Debugger debugger;
  auto model = std::make_unique<mtr::TrtMTR>(model_path);

  debugger.createEvent();
  auto config = model->config();
  auto agent_data = loadAgentData();
  auto polyline_data =
    loadPolylineData(config.max_num_polyline, config.max_num_point, config.point_break_distance);
  debugger.printElapsedTime("Data loading time: ");

  for (int i = 0; i < num_repeat; ++i) {
    debugger.createEvent();
    std::vector<mtr::PredictedTrajectory> trajectories;
    if (!model->doInference(agent_data, polyline_data, trajectories)) {
      std::cerr << "===== [FAIL]: Fail to inference!! =====" << std::endl;
    } else {
      std::cout << "===== [SUCCESS] Success to inference!! =====" << std::endl;
    }
    debugger.printElapsedTime("Inference time: ");
  }
}
