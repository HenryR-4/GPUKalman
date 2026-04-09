add_library(LibraryBasedMatrixOperations LibraryBasedMatrixOperationsImpl.cu)
target_include_directories(LibraryBasedMatrixOperations PUBLIC ../include)
target_link_libraries(LibraryBasedMatrixOperations PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver)

add_library(devicetimer DeviceTimer.cpp)
target_include_directories(devicetimer PUBLIC ../include) 
target_link_libraries(devicetimer CUDA::cudart)
