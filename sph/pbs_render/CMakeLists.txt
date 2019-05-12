#pbs_render config
cmake_minimum_required(VERSION 3.2.2)

set (PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set (PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set (PROJECT_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shader)

include_directories("${PROJECT_INCLUDE_DIR}")

include_directories("${GLOBAL_DIR}/extlib/include")
link_directories("${GLOBAL_DIR}/extlib/lib")

include_directories("${LIBRARY_OUTPUT_PATH}")
file(GLOB_RECURSE SPHRENDER_FILES
	"${PROJECT_INCLUDE_DIR}/*.h"
	"${PROJECT_SOURCE_DIR}/*.cpp"
)

file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/shaders DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/models DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
add_executable(pbs_render ${SPHRENDER_FILES})
target_link_libraries(pbs_render pbs_mathlib opengl32 glew32s glfw3)
