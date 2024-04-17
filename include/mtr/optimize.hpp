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

#ifndef MTR__OPTIMIZE_HPP_
#define MTR__OPTIMIZE_HPP_

#include <NvInfer.h>

namespace mtr
{
void createOptimizationProfile(
  nvinfer1::IBuilder * builder, nvinfer1::INetworkDefinition * network,
  nvinfer1::IBuilderConfig * config)
{
  // input
  auto nbInputs = network->getNbInputs();
  for (int32_t i = 0; i < nbInputs; ++i) {
    auto inTensor = network->getInput(i);
    auto name = inTensor->getName();
    auto profile = builder->createOptimizationProfile();
    // profile->setDimensions();
  }
}

void addInput(
  nvinfer1::INetworkDefinition * network, const char * name, nvinfer1::DataType type,
  nvinfer1::Dims dimensions)
{
  auto inType = network->addInput(name, type, dimensions);
}
}  // namespace mtr

#endif  // MTR__OPTIMIZE_HPP_
