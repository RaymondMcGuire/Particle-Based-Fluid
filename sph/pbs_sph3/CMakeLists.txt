#pbs_sph3 config
cmake_minimum_required(VERSION 3.2.2)

set (PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set (PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories("${PROJECT_INCLUDE_DIR}")
include_directories("${LIBRARY_OUTPUT_PATH}")
file(GLOB_RECURSE SPH3_FILES
	"${PROJECT_INCLUDE_DIR}/*.h"
	"${PROJECT_SOURCE_DIR}/*.cpp"
)

add_executable(pbs_sph3 ${SPH3_FILES})
target_link_libraries(pbs_sph3 pbs_mathlib)