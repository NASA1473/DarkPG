
cmake_minimum_required(VERSION 3.12)
project( OR_and_PGS_in_car_park_based_on_FT-2000/4_Processor )

set( CMAKE_CXX_STANDARD 14)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
endif()

set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )  

find_package( OpenCV REQUIRED )
find_package( MPI REQUIRED )

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OpenMP found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    ${MPI_INCLUDE_PATH}
    /home/kylin/fftw/include/
)

link_directories(/home/kylin/fftw/lib/)

link_libraries(fftw3f)
link_libraries(m)

include_directories( ${PROJECT_SOURCE_DIR}/include )

add_subdirectory( src )
add_subdirectory( test )

set( CMAKE_CXX_COMPILER mpicxx )
set( CMAKE_C_COMPILER mpicc )







