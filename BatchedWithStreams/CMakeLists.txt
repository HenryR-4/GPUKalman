cmake_minimum_required(VERSION 3.12)

project(KalmanBatched LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(SOURCES
    src/main.cu
    src/pipe.cu
)

add_executable(test ${SOURCES})
target_include_directories(test PRIVATE include)
target_link_libraries(test PRIVATE CUDA::cudart CUDA::cublas)
