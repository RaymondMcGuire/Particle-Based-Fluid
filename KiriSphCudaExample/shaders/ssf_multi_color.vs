#version 330 core
layout(location = 0) in vec4 aPos;
layout(location = 1) in vec4 aColor;

layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
};

uniform float particleScale;
uniform float particleSize;
uniform bool transparent;

out VS_OUT { vec4 color; }
vs_out;

void main() {
  mat4 model = mat4(1.0f);

  vs_out.color = aColor;

  vec3 cameraSpacePos = (view * model * vec4(aPos.xyz, 1.0f)).xyz;

  if (transparent && aColor.w == 0.0)
    gl_PointSize = -particleScale * 0.f / cameraSpacePos.z;
  else
    gl_PointSize = -particleScale * aPos.w / cameraSpacePos.z;

  gl_Position = projection * view * model * vec4(aPos.xyz, 1.0f);
}