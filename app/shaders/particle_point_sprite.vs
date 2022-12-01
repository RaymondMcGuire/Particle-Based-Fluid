#version 330 core
layout(location = 0) in vec4 aPos;

layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
};

out vec3 cameraSpacePos;

uniform float particleScale;
uniform float particleSize;

void main() {
  mat4 model = mat4(1.0f);
  cameraSpacePos = (view * model * vec4(aPos.xyz, 1.0f)).xyz;
  gl_PointSize = -particleScale * aPos.w / cameraSpacePos.z;
  gl_Position = projection * view * model * vec4(aPos.xyz, 1.0f);
}