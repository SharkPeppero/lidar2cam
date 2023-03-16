cmake_minimum_required(VERSION 2.8)
project(lidar2cam)

set(CMAKE_BUILD_TYPE "Debug")
set( CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall" )

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

find_package( PCL REQUIRED )
include_directories(${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories("/usr/include/eigen3")

add_executable( lidar2cam_node  lidar2cam.cpp )
# 与Ceres和OpenCV链接
target_link_libraries( lidar2cam_node ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})