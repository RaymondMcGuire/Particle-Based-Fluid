#version 330 core

layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
};

in VS_OUT { vec4 color; }
fs_in;

out vec4 FragColor;

void main() {

  vec3 N;
  N.xy = gl_PointCoord.xy * vec2(2.0, 2.0) + vec2(-1.0, -1.0);

  float mag = dot(N.xy, N.xy);
  if (mag > 1.0) {
    discard;
    return;
  }

  N.z = sqrt(1.0 - mag);

  FragColor = fs_in.color;
}