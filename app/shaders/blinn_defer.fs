#version 330 core
out vec4 FragColor;

layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
};

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
uniform sampler2D ssao;

struct Light {
  vec3 Position;
  vec3 Color;

  float Linear;
  float Quadratic;
  float Radius;
};
const int NR_LIGHTS = 32;
uniform Light lights[NR_LIGHTS];

uniform bool b_ssao;

void main() {
  vec3 FragPos = texture(gPosition, TexCoords).rgb;
  vec3 Normal = texture(gNormal, TexCoords).rgb;
  vec3 Diffuse = texture(gAlbedoSpec, TexCoords).rgb;
  float Specular = texture(gAlbedoSpec, TexCoords).a;
  float AmbientOcclusion = 1.0;

  if (b_ssao) {
    AmbientOcclusion = texture(ssao, TexCoords).r;
  }

  // calculate lighting
  vec3 ambient = vec3(0.9 * Diffuse * AmbientOcclusion);
  vec3 lighting = ambient;
  // vec3 viewDir  = normalize(viewPos - FragPos);
  // cam pos in view-space is (0.0.0)
  vec3 viewDir = normalize(-FragPos);
  for (int i = 0; i < NR_LIGHTS; ++i) {
    vec3 lightPos = vec3(view * vec4(lights[i].Position, 1.0));
    // not optimize
    // calculate distance between light source and current fragment
    float distance = length(lightPos - FragPos);
    if (distance < lights[i].Radius) {
      // diffuse
      vec3 lightDir = normalize(lightPos - FragPos);
      vec3 diffuse =
          max(dot(Normal, lightDir), 0.0) * Diffuse * lights[i].Color;
      // specular
      vec3 halfwayDir = normalize(lightDir + viewDir);
      float spec = pow(max(dot(Normal, halfwayDir), 0.0), 16.0);
      vec3 specular = lights[i].Color * spec * Specular;
      // attenuation
      float distance = length(lightPos - FragPos);
      float attenuation = 1.0 / (1.0 + lights[i].Linear * distance +
                                 lights[i].Quadratic * distance * distance);
      diffuse *= attenuation;
      specular *= attenuation;
      lighting += diffuse + specular;
    }
  }
  FragColor = vec4(lighting, 1.0);
}
