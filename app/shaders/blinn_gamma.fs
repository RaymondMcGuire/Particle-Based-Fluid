#version 330 core
out vec4 FragColor;

layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
};

struct Material {
  sampler2D diffuse;
  sampler2D specular;
  sampler2D normal;
  float shininess;
};

struct PointLight {
  vec3 position;

  float constant;
  float linear;
  float quadratic;

  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
};

in VS_OUT {
  vec3 FragPos;
  vec3 Normal;
  vec2 TexCoords;
}
fs_in;

#define NR_POINT_LIGHTS 10
uniform PointLight pointLights[NR_POINT_LIGHTS];

uniform Material material;

uniform bool gamma;
uniform int pointLightNum;

vec3 BlinnPhong(PointLight light, vec3 normal) {
  // ambient
  vec3 ambient =
      light.ambient * vec3(texture(material.diffuse, fs_in.TexCoords));
  ;

  // diffuse
  vec3 lightDir = normalize(light.position - fs_in.FragPos);
  float diff = max(dot(lightDir, normal), 0.0);
  vec3 diffuse =
      diff * light.diffuse * vec3(texture(material.diffuse, fs_in.TexCoords));

  // specular
  vec3 viewDir = normalize(camPos - fs_in.FragPos);
  vec3 reflectDir = reflect(-lightDir, normal);
  float spec = 0.0;
  vec3 halfwayDir = normalize(lightDir + viewDir);
  spec = pow(max(dot(normal, halfwayDir), 0.0), material.shininess);
  vec3 specular = spec * light.specular;

  float distance = length(light.position - fs_in.FragPos);
  float attenuation = 1.0 / (light.constant + light.linear * distance +
                             light.quadratic * (distance * distance));

  ambient *= attenuation;
  diffuse *= attenuation;
  specular *= attenuation;

  return (ambient / pointLightNum + diffuse + specular);
}

void main() {

  vec3 lighting = vec3(0.0);
  for (int i = 0; i < pointLightNum; ++i)
    lighting += BlinnPhong(pointLights[i], normalize(fs_in.Normal));

  if (gamma)
    lighting = pow(lighting, vec3(1.0 / 2.2));
  FragColor = vec4(lighting, 1.0);
}