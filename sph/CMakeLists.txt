cmake_minimum_required(VERSION 2.8)
project(demo_sph CXX)

include_directories(
	"${PROJECT_SOURCE_DIR}/include"
)

file(GLOB_RECURSE sph_FILES
	"${PROJECT_SOURCE_DIR}/include/*.h"
	"${PROJECT_SOURCE_DIR}/src/*.cpp"
)

add_executable(demo_sph ${sph_FILES})