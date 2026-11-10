cmake_minimum_required(VERSION 3.16)
project(workspace_reconstruction C CXX)  # C is needed for the MPI C target below

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Ubuntu's VTK package (pulled in by PCL) references MPI targets; finding MPI first avoids a CMake error
find_package(MPI QUIET COMPONENTS C)
find_package(PCL 1.10 REQUIRED COMPONENTS common io filters search segmentation sample_consensus)

add_executable(workspace_segmentation src/workspace_segmentation.cpp)
target_include_directories(workspace_segmentation PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(workspace_segmentation PRIVATE ${PCL_LIBRARIES})
target_compile_definitions(workspace_segmentation PRIVATE ${PCL_DEFINITIONS})
