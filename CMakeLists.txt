cmake_minimum_required(VERSION 3.17)

# project name
project(CUDA_CMAKE LANGUAGES CXX CUDA)

# All C++ files except main.cc and bench.cc
set(CXX_SRC src/lbp.hh)

# All cuda files
set(CUDA_SRC src/lbp.hh src/lbp.cu src/lbp2.cu src/lbp3.cu src/nn.cu src/nn2.cu)

# find dependencies
find_package(OpenCV REQUIRED)

# build a shared library with our CUDA code
add_library(gpuproc SHARED ${CUDA_SRC})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -O0 -DDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")

target_compile_features(gpuproc PUBLIC cxx_std_17)
set_target_properties(gpuproc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpuproc)

# build the C++ code and link with the CUDA code
add_executable(lbp src/main.cc ${CXX_SRC})

target_link_libraries(lbp gpuproc
  opencv_core
  opencv_video
  opencv_videoio
  opencv_highgui
)

include(FetchContent)

FetchContent_Declare(
  GoogleBenchmark
  URL https://github.com/google/benchmark/archive/v1.5.4.tar.gz
)

if (NOT GoogleBenchmark_POPULATED)
  FetchContent_Populate(GoogleBenchmark)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
  add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

add_executable(bench ${SRC} bench_src/bench.cc)
target_compile_features(bench PUBLIC cxx_std_17)
target_include_directories(bench PUBLIC ${googlebenchmark_SOURCE_DIR}/include)
target_include_directories(bench PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(bench gpuproc
  opencv_core
  opencv_video
  opencv_videoio
  opencv_highgui
  benchmark
)

if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
	message("                                 ")
	message("             ^                   ")
	message("            / \\                  ")
	message("           / | \\                 ")
	message("          /  |  \\                ")
	message("         /   .   \\               ")
	message("        /_________\\              ")
	message("                                 ")
 	message("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
	message("@ Warning: build in debug mode, @")
	message("@ benchmarks will be inaccurate @")
 	message("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
endif()


