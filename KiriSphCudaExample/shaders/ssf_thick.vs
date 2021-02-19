#version 330 core
layout(location = 0) in vec4 aPos;
layout(location = 1) in vec4 aColor;
layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
};

out float particleRadius;

uniform float particleScale;
uniform float particleSize;

void main() {
  mat4 model = mat4(1.0f);
  vec3 cameraSpacePos = (view * model * vec4(aPos.xyz, 1.0f)).xyz;

  particleRadius = aPos.w;
  gl_PointSize = -particleScale * particleRadius / cameraSpacePos.z;
  gl_Position = projection * view * model * vec4(aPos.xyz, 1.0f);
}