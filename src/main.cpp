#include "mtr/mtr.hpp"

#include <cassert>
#include <string>

static const std::string USAGE = "[USAGE]: ./build/main <PATH_TO_ONNX> [or <PATH_TO_ENGINE>]";

int main(int argc, char ** argv)
{
  assert((USAGE, argc == 2));
  auto model_path = std::string(argv[1]);

  auto model = std::make_unique<mtr::TrtMTR>(model_path, "FP32");
}