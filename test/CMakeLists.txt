add_executable( MAIN_MPI MAIN_MPI.cpp)
add_executable( MAIN_NEON_MPI MAIN_neon_MPI.cpp)
add_executable( MAIN_ALONE MAIN_ALONE.cpp)
target_link_libraries( MAIN_MPI 
    lime_feature
    yolo_feature
    inference_feature
    ${OpenCV_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    MPI::MPI_CXX
)

target_link_libraries( MAIN_NEON_MPI 
    lime_feature
    yolo_feature
    inference_feature
    ${OpenCV_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    MPI::MPI_CXX
)

target_link_libraries( MAIN_ALONE 
    lime_feature
    yolo_feature
    inference_feature
    ${OpenCV_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
)
