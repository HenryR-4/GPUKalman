cmake_minimum_required(VERSION 3.12)

project(KalmanFilterGPGPU LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(tests)
add_subdirectory(src)
add_subdirectory(examples)
