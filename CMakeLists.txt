cmake_minimum_required(VERSION 2.8)
project( mono-vo )
find_package( OpenCV 3 REQUIRED )
find_package(OpenMP)

include_directories( ${OpenCV_INCLUDE_DIRS} )
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

file(GLOB viso
    "src/*.h"
    "src/*.cpp"
)
add_executable( vo ${viso} )
target_link_libraries( vo ${OpenCV_LIBS} )
