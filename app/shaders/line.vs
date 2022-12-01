#version 330 core
layout (location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
};
uniform mat4 model;

out VS_OUT { vec3 color; }
vs_out;

void main()
{
  vs_out.color = aColor;
  gl_Position = projection * view * model * vec4(aPos, 1.0);
}