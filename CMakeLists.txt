cmake_minimum_required(VERSION 3.14)
project(trt_mtr)

# For CUDA
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
  message("CUDA_LIBRARIES: " ${CUDA_LIBRARIES})
  message("CUBLAS_LIBRARIES: " ${CUBLAS_LIBRARIES})
  message("CUDNN_LIBRARIES: " ${CUDNN_LIBRARIES})
  message("CUDA_npp_LIBRARY: " ${CUDA_npp_LIBRARY})
  message("CUDA_nppc_LIBRARY: " ${CUDA_nppc_LIBRARY})
  message("CUDA_npps_LIBRARY: " ${CUDA_npps_LIBRARY})
else()
  message(FATAL_ERROR "Can not find CUDA")
endif()

# For TensorRT
list(APPEND TRT_PLUGINS "nvinfer")
list(APPEND TRT_PLUGINS "nvonnxparser")
list(APPEND TRT_PLUGINS "nvparsers")

foreach(libName ${TRT_PLUGINS})
  find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIES lib)
  list(APPEND TRT_PLUGIN_LIBS ${${libName}_lib})
endforeach()
message("TRT_PLUGIN_LIBS: " ${TRT_PLUGIN_LIBS})

cuda_add_library(attention_plugin SHARED
  lib/src/attention/trt_attn_value_computation_kernel.cu
  lib/src/attention/trt_attn_value_computation.cpp
  lib/src/attention/trt_attn_weight_computation_kernel.cu
  lib/src/attention/trt_attn_weight_computation.cpp
)

cuda_add_library(knn_plugin SHARED 
  lib/src/knn/trt_knn_batch_kernel.cu
  lib/src/knn/trt_knn_batch.cpp
  lib/src/knn/trt_knn_batch_mlogk_kernel.cu
  lib/src/knn/trt_knn_batch_mlogk.cpp)

target_link_libraries(attention_plugin
    ${TRT_PLUGINS}
    ${TRT_PLUGIN_LIBS}
    ${CUDA_LIBRARIES}
    ${CUBLAS_LIBRARIES}
    ${CUDNN_LIBRARIES}
    ${CUDA_npp_LIBRARY}
    ${CUDA_nppc_LIBRARY}
    ${CUDA_npps_LIBRARY})

target_link_libraries(knn_plugin
    ${TRT_PLUGINS}
    ${TRT_PLUGIN_LIBS}
    ${CUDA_LIBRARIES}
    ${CUBLAS_LIBRARIES}
    ${CUDNN_LIBRARIES}
    ${CUDA_npp_LIBRARY}
    ${CUDA_nppc_LIBRARY}
    ${CUDA_npps_LIBRARY})  

target_include_directories(attention_plugin PUBLIC
  lib/include
)

target_include_directories(knn_plugin PUBLIC
  lib/include
)

add_library(trt_mtr SHARED src/mtr.cpp src/builder.cpp)
target_link_libraries(trt_mtr attention_plugin knn_plugin)

target_include_directories(trt_mtr PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(main src/main.cpp)
target_link_libraries(main PUBLIC trt_mtr)