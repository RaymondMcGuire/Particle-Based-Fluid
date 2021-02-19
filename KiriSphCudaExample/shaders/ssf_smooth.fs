#version 420
#extension GL_EXT_shader_image_load_store : enable

uniform int kernelR;
uniform float sigmaR;
uniform float sigmaZ;

uniform int blurOption;
uniform bool particleView;
uniform bool enableSSF;

uniform sampler2D multiColorTex;

/* zA: source depth map, zB: target depth map */
uniform sampler2D zA;
uniform layout(r32f) image2D zB;

in VS_OUT { vec2 texCoord; }
fs_in;

void setZ(int x, int y, float z) {
  imageStore(zB, ivec2(x, y), vec4(z, 0, 0, 0));
}

float getZ(int x, int y) { return texelFetch(zA, ivec2(x, y), 0).x; }

float bilateral(int x, int y) {
  float z = getZ(x, y);
  float sum = 0, wsum = 0;

  for (int dx = -kernelR; dx <= kernelR; dx++)
    for (int dy = -kernelR; dy <= kernelR; dy++) {
      float s = getZ(x + dx, y + dy);

      float w = exp(-(dx * dx + dy * dy) * sigmaR * sigmaR);

      float r2 = (s - z) * sigmaZ;
      float g = exp(-r2 * r2);

      float wg = w * g;
      sum += s * wg;
      wsum += wg;
    }

  if (wsum > 0)
    sum /= wsum;
  return sum;
}

float gaussian(int x, int y) {
  float z = getZ(x, y);
  float sum = 0, wsum = 0;

  for (int dx = -kernelR; dx <= kernelR; dx++)
    for (int dy = -kernelR; dy <= kernelR; dy++) {
      float s = getZ(x + dx, y + dy);
      float w = exp(-(dx * dx + dy * dy) * sigmaR * sigmaR);

      sum += s * w;
      wsum += w;
    }

  if (wsum > 0)
    sum /= wsum;
  return sum;
}

void main() {

  int x = int(gl_FragCoord.x), y = int(gl_FragCoord.y);

  if (!enableSSF) {
    // for solid, need optimize
    float w = texture(multiColorTex, fs_in.texCoord).w;
    if (abs(w - 0.5) < 1e-6)
      return;
  } else {
    if (particleView)
      return;
  }

  float z = getZ(x, y);
  if (z > 99)
    return;

  float zz;
  if (blurOption == 0)
    zz = bilateral(x, y);
  else
    zz = gaussian(x, y);
  setZ(x, y, zz);
}