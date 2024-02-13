#include "preprocess/agent_preprocess_kernel.cuh"
#include "preprocess/preprocess_kernel.hpp"

#include <iostream>

cudaError_t polylinePreprocessLauncher(
  const float * polylines, float * out_polylines, bool * out_mask)
{
  return cudaGetLastError();
}
