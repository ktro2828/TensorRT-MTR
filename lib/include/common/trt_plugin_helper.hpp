#ifndef TRT_PLUGIN_HELPER_HPP
#define TRT_PLUGIN_HELPER_HPP

#include <NvInferRuntime.h>

#include <iostream>

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

enum pluginStatus_t {
  STATUS_SUCCESS = 0,
  STATUS_FAILURE = 1,
  STATUS_BAD_PARAM = 2,
  STATUS_NOT_SUPPORTED = 3,
  STATUS_NOT_INITIALIZED = 4
};  // enum pluginStatus_t

#define ASSERT(assertion)                                                    \
  {                                                                          \
    if (!(assertion)) {                                                      \
      std::cerr << "#assertion" << __FILE__ << "," << __LINE__ << std::endl; \
      abort();                                                               \
    }                                                                        \
  }

#endif  // TRT_PLUGIN_HELPER_HPP