#version 330 core

layout(location = 0) out vec4 normalTex;

in vec2 texCoord;

uniform float particleScale;
uniform float screenWidth;
uniform float screenHeight;

uniform bool keepEdge;

uniform sampler2D depthTex;

/* return z in right-hand coord
 * Because both opengl and paper assume right-hand coord
 * store as left-hand only for easy debug
 */
float getZ(float x, float y) { return -texture(depthTex, vec2(x, y)).x; }

void main() {
  /* global */
  float c_x = 2 / (particleScale);
  float c_y = 2 / (particleScale);

  /* (x, y) in [0, 1] */
  float x = texCoord.x, y = texCoord.y;
  float dx = 1 / screenWidth, dy = 1 / screenHeight;
  float z = getZ(x, y), z2 = z * z;
  float dzdx = getZ(x + dx, y) - z, dzdy = getZ(x, y + dy) - z;
  float dzdx2 = z - getZ(x - dx, y), dzdy2 = z - getZ(x, y - dy);

  /* Skip silhouette */
  if (keepEdge) {
    if (abs(dzdx2) < abs(dzdx))
      dzdx = dzdx2;
    if (abs(dzdy2) < abs(dzdy))
      dzdy = dzdy2;
  }

  vec3 n = vec3(-c_y * dzdx, -c_x * dzdy, c_x * c_y * z);
  /* revert n.z to positive for debugging */
  n.z = -n.z;

  float d = length(n);
  normalTex = vec4(n / d, d);
}