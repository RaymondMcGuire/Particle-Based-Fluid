#version 330 core
layout(location = 0) out float thickness;

uniform float particleSize;

in float particleRadius;
void main() {

  vec3 normal;
  normal.xy = gl_PointCoord.xy * vec2(2.0, 2.0) + vec2(-1.0, -1.0);
  float mag = dot(normal.xy, normal.xy);
  if (mag > 1.0) {
    discard;
    return;
  }

  normal.z = sqrt(1.0 - mag);

  vec3 lightDir = vec3(0, 0, 1);
  thickness = 2 * particleRadius * dot(normal, lightDir);
}