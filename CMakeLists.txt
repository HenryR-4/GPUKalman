cmake_minimum_required(VERSION 3.12)

project(KalmanFilterGPGPU LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# find_library(CUBLAS_PATH 
#     NAMES cublas
#     HINTS "${CUBLAS_ROOT}"
#     ENV CUBLAS_ROOT
#     PATH_SUFFIXES lib lib64
# )
# 
# if(NOT CUBLAS_PATH)
#     message(FATAL_ERROR "Could not find cublas, set CUBLAS_ROOT or CMAKE_PREFIX_PATH")
# endif()
# 
# add_library(cublas UNKNOWN IMPORTED)
# set_target_properties(cublas PROPERTIES
#     IMPORTED_LOCATION "${CUBLAS_PATH}"
# )
# 
# find_library(CUSOLVER_PATH 
#     NAMES cusolver 
#     HINTS "${CUSOLVER_ROOT}"
#     ENV CUSOLVER_ROOT
#     PATH_SUFFIXES lib lib64
# )
# 
# if(NOT CUSOLVER_PATH)
#     message(FATAL_ERROR "Could not find cusolver, set CUSOLVER_ROOT or CMAKE_PREFIX_PATH")
# endif()
# 
# add_library(cusolver UNKNOWN IMPORTED)
# set_target_properties(cusolver PROPERTIES
#     IMPORTED_LOCATION "${CUSOLVER_PATH}"
# )

add_subdirectory(tests)
add_subdirectory(src)
add_subdirectory(examples)
