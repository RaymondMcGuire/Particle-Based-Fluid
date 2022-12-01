#version 330 core
out vec4 FragColor;
in VS_OUT {
  vec3 FragPos;
  vec3 Normal;
  flat int Instance;
}
fs_in;

uniform bool singleColor;
uniform vec3 dirLightPos;
uniform vec3 lightColor;
uniform vec3 particleColor;

uniform samplerBuffer color_tbo;

void main() {
  // ambient
  float ambientStrength = 0.1;
  vec3 ambient = ambientStrength * lightColor;

  // diffuse
  vec3 norm = normalize(fs_in.Normal);
  vec3 lightDir = normalize(dirLightPos - fs_in.FragPos);
  float diff = max(dot(norm, lightDir), 0.0);
  vec3 diffuse = diff * lightColor;

  vec3 color = (ambient + diffuse);
  if (singleColor) {
    color *= particleColor;
  } else {
    color *= texelFetch(color_tbo, fs_in.Instance).rgb;
  }

  FragColor = vec4(color, 1.0);
}