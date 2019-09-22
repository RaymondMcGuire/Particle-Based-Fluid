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

#OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#subproject setting
add_subdirectory(pbs_mathlib)
add_subdirectory(pbs_sph3)
add_subdirectory(pbs_render)
add_subdirectory(pbs_model)