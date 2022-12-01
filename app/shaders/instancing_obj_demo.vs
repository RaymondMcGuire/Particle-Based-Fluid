#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 2) in vec2 aTexCoords;
layout(location = 3) in mat4 aInstanceMatrix;

out vec2 TexCoords;

layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
};

void main() {
  TexCoords = aTexCoords;
  gl_Position = projection * view * aInstanceMatrix * vec4(aPos, 1.0f);
}