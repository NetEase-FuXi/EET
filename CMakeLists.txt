cmake_minimum_required(VERSION 3.1)

project(eet VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libtorch")
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#set(CUDA_NVCC_FLAGS "-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__")
include_directories(csrc)
include_directories(${Python_INCLUDE_DIRS})

aux_source_directory(csrc/op source_op)
aux_source_directory(csrc/core source_core)
cuda_add_library(eet SHARED ${source_op} ${source_core} small_test/latency_test.cpp)
set_property(TARGET eet PROPERTY CUDA_ARCHITECTURES 72)
target_link_libraries(eet ${TORCH_LIBRARIES})
target_link_libraries(eet ${CUDA_LIBRARIES})

aux_source_directory(example/cpp test_src)
cuda_add_executable(param_counter ${test_src})
target_link_libraries(param_counter eet)
