#version 330 core

layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
};

uniform float particleSize;

in float particleRadius;
in vec3 cameraSpacePos;

out vec4 FragColor;

void main() {

  vec3 normal;
  normal.xy = gl_PointCoord.xy * vec2(2.0, 2.0) + vec2(-1.0, -1.0);

  float mag = dot(normal.xy, normal.xy);
  if (mag > 1.0) {
    discard;
    return;
  }

  normal.z = sqrt(1.0 - mag);

  vec4 pixelEyePos = vec4(cameraSpacePos + normal * particleRadius, 1.0f);
  vec4 pixelClipPos = projection * pixelEyePos;
  float ndcZ = pixelClipPos.z / pixelClipPos.w;
  gl_FragDepth = 0.5 * (gl_DepthRange.diff * ndcZ + gl_DepthRange.far +
                        gl_DepthRange.near);

  // gl_FragCoord.z is NOT projPos.x / projPos.w!
  // Write to d_depth_r
  FragColor.r = -pixelEyePos.z;
}