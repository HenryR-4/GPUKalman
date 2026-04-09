add_executable(test_Matrix test_Matrix.cpp)
target_include_directories(test_Matrix PRIVATE ../include)
target_link_libraries(test_Matrix PRIVATE CUDA::cudart)

add_executable(test_KalmanFilter test_KalmanFilter.cpp)
target_link_libraries(test_KalmanFilter PRIVATE LibraryBasedMatrixOperations)

add_executable(test_LibraryBasedMatrixOperations test_MatrixOperations.cpp)
target_link_libraries(test_LibraryBasedMatrixOperations PRIVATE LibraryBasedMatrixOperations)

enable_testing()

add_test(NAME Matrix_test COMMAND $<TARGET_FILE:test_Matrix>)
add_test(NAME KalmanFilter_test COMMAND $<TARGET_FILE:test_KalmanFilter>)
add_test(NAME LibraryBasedMatrixOperations_test COMMAND $<TARGET_FILE:test_LibraryBasedMatrixOperations>)
