cmake_minimum_required(VERSION 3.14)
project(trt_mtr)

# -------- default to C++17 --------
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# -------- create compile_commands.json --------
if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# -------- find cuda libraries --------
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
  message("Cuda version: " ${CUDA_VERSION})
  include_directories(${CUDA_INCLUDE_DIRS})
  find_library(CUBLAS_LIBRARIES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                                             ${CUDA_TOOLKIT_ROOT_DIR}/lib)
  find_library(
    CUDNN_LIBRARIES
    NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib
          ${__cudnn_lib_win_name}
    PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR}
          ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
    PATH_SUFFIXES lib lib64 bin
    DOC "CUDNN library.")
else()
  message(FATAL_ERROR "Can not find CUDA")
endif()

# -------- find tensorrt libraries --------
list(APPEND TRT_PLUGINS "nvinfer")
list(APPEND TRT_PLUGINS "nvonnxparser")
list(APPEND TRT_PLUGINS "nvparsers")

foreach(libName ${TRT_PLUGINS})
  find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIES lib)
  list(APPEND TRT_PLUGIN_LIBS ${${libName}_lib})
endforeach()

# -------- link targets for custom tensorrt plugins --------
file(GLOB_RECURSE CUSTOM_PLUGIN_FILES lib/src/attention/* lib/src/knn/*)
cuda_add_library(custom_plugin SHARED ${CUSTOM_PLUGIN_FILES})
target_link_libraries(custom_plugin
    ${TRT_PLUGINS}
    ${TRT_PLUGIN_LIBS}
    ${CUDA_LIBRARIES}
    ${CUBLAS_LIBRARIES}
    ${CUDNN_LIBRARIES})
target_include_directories(custom_plugin PUBLIC
  lib/include
)

# -------- link targets for custom preprocess/postprocess kernels --------
file(GLOB_RECURSE CUSTOM_KERNEL_FILES lib/src/preprocess/* lib/src/postprocess/*)
cuda_add_library(custom_kernel SHARED ${CUSTOM_KERNEL_FILES})
target_link_libraries(custom_kernel
  ${CUDA_LIBRARIES}
  ${CUBLAS_LIBRARIES}
  ${CUDNN_LIBRARIES})
target_include_directories(custom_kernel PUBLIC
  lib/include
  ${PROJECT_SOURCE_DIR}/include
)

# -------- link targets for mtr --------
add_library(trt_mtr SHARED src/mtr.cpp src/builder.cpp)
target_link_libraries(trt_mtr custom_plugin custom_kernel)
target_include_directories(trt_mtr PUBLIC ${PROJECT_SOURCE_DIR}/include)

# -------- add executable --------
add_executable(main src/main.cpp)
target_link_libraries(main PUBLIC trt_mtr)

# -------- for testing --------
add_executable(test_agent test/agent_test.cpp)
add_executable(test_polyline test/polyline_test.cpp)
add_executable(test_intention_point test/intention_point_test.cpp)
target_include_directories(test_agent PUBLIC include)
target_include_directories(test_polyline PUBLIC include)
target_include_directories(test_intention_point PUBLIC include)
