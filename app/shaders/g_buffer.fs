#version 330 core
layout(location = 0) out vec3 gPosition;
layout(location = 1) out vec3 gNormal;
layout(location = 2) out vec4 gAlbedoSpec;

in VS_OUT {
  vec3 FragPos;
  vec2 TexCoords;
  vec3 Normal;
  mat3 TBN;
}
fs_in;

uniform sampler2D texture_diffuse1;
uniform sampler2D texture_specular1;
uniform sampler2D texture_normal1;

uniform bool use_normal;
void main() {
  // position buffer
  gPosition = fs_in.FragPos;

  // normal buffer
  if (use_normal) {
    // normal buffer (use normal map)
    vec3 norm = vec3(texture(texture_normal1, fs_in.TexCoords)).rgb;
    // tangent space
    norm = normalize(norm * 2.0 - 1.0);
    // world space
    gNormal = transpose(inverse(fs_in.TBN)) * norm;
  } else {
    gNormal = normalize(fs_in.Normal);
  }

  // diffuse buff
  gAlbedoSpec.rgb = texture(texture_diffuse1, fs_in.TexCoords).rgb;
  // spec buffer
  gAlbedoSpec.a = texture(texture_specular1, fs_in.TexCoords).r;
}