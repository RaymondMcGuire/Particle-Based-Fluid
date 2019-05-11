#Particle Based Simulation Project Settings
cmake_minimum_required(VERSION 3.2.2)
project(pbs C CXX)
set(CMAKE_BUILD_TYPE Debug)

#binary dir/ output file
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/out)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

#global include
set(GLOBAL_DIR ${PROJECT_SOURCE_DIR})

#subproject setting
add_subdirectory(pbsMathLib)
add_subdirectory(pbs_sph3)
add_subdirectory(pbs_render)
add_subdirectory(pbs_model)