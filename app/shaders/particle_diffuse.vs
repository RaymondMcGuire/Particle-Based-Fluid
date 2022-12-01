#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
};

out VS_OUT {
  vec3 FragPos;
  vec3 Normal;
}
vs_out;

uniform mat4 model;

void main() {
  vs_out.FragPos = aPos;
  vs_out.Normal = aNormal;
  gl_Position = projection * view * model * vec4(aPos, 1.0);
}