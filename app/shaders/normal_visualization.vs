#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
};

out VS_OUT { vec3 normal; }
vs_out;

uniform mat4 model;

void main() {
  mat3 normalMatrix = mat3(transpose(inverse(view * model)));
  vs_out.normal = vec3(projection * vec4(normalMatrix * aNormal, 0.0));
  gl_Position = projection * view * model * vec4(aPos, 1.0);
}