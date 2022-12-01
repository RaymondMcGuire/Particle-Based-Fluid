#version 330 core
layout(location = 0) in vec3 aPos;

layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
};

out vec3 WorldPos;

void main() {
  WorldPos = aPos;

  mat4 rotView = mat4(mat3(view));
  vec4 clipPos = projection * rotView * vec4(WorldPos, 1.0);

  gl_Position = clipPos.xyww;
}