#version 330 core
layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 BrightColor;

layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
};

struct MaterialTex {
  sampler2D diffuse;
  sampler2D specular;
  sampler2D normal;
  float shininess;
};

struct Material {
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
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

struct TextureMap {
  bool enable;
  bool diffuse;
  bool specular;
  bool normal;
};

in VS_OUT {
  vec3 FragPos;
  vec2 TexCoords;
  vec3 Normal;
  mat3 TBN;
  vec3 TangentViewPos;
  vec3 TangentFragPos;
}
fs_in;

#define MAX_POINT_LIGHTS 10
uniform PointLight pointLights[MAX_POINT_LIGHTS];
uniform MaterialTex materialTex;
uniform Material material;
uniform TextureMap textureMap;

uniform bool gamma;
uniform int pointLightNum;
// function prototypes
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir);

void main() {
  vec3 norm, _viewPos, _fragPos;
  if (textureMap.normal) {
    // obtain normal from normal map in range [0,1]
    norm = vec3(texture(materialTex.normal, fs_in.TexCoords));
    // transform normal vector to range [-1,1]
    norm = normalize(norm * 2.0 - 1.0); // this normal is in tangent space

    _viewPos = fs_in.TangentViewPos;
    _fragPos = fs_in.TangentFragPos;
  } else {
    norm = normalize(fs_in.Normal);
    _viewPos = camPos;
    _fragPos = fs_in.FragPos;
  }

  vec3 viewDir = normalize(_viewPos - _fragPos);

  vec3 result = vec3(0.0);
  for (int i = 0; i < pointLightNum; i++)
    result += CalcPointLight(pointLights[i], norm, _fragPos, viewDir);

  float brightness = dot(result, vec3(0.2126, 0.7152, 0.0722));
  if (brightness > 1.0)
    BrightColor = vec4(result, 1.0);
  else
    BrightColor = vec4(0.0, 0.0, 0.0, 1.0);

  if (gamma)
    result = pow(result, vec3(1.0 / 2.2));
  FragColor = vec4(result, 1.0);
}

// point light
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
  vec3 _lightPos = light.position;
  if (textureMap.normal) {
    _lightPos = fs_in.TBN * _lightPos;
  }

  vec3 lightDir = normalize(_lightPos - fragPos);
  // diffuse shading
  float diff = max(dot(normal, lightDir), 0.0);

  // attenuation
  float distance = length(light.position - fs_in.FragPos);
  float attenuation = 1.0 / (light.constant + light.linear * distance +
                             light.quadratic * (distance * distance));

  // blinn-phong
  vec3 ambient = vec3(0.0);
  vec3 diffuse = vec3(0.0);
  vec3 specular = vec3(0.0);
  if (textureMap.enable) {
    ambient =
        light.ambient * vec3(texture(materialTex.diffuse, fs_in.TexCoords));
    diffuse = light.diffuse * diff *
              vec3(texture(materialTex.diffuse, fs_in.TexCoords));

    if (textureMap.specular) {
      // specular shading
      vec3 reflectDir = reflect(-lightDir, normal);
      vec3 halfwayDir = normalize(lightDir + viewDir);
      float spec =
          pow(max(dot(normal, halfwayDir), 0.0), materialTex.shininess);
      specular = light.specular * spec *
                 vec3(texture(materialTex.specular, fs_in.TexCoords));
    }
  } else {
    ambient = light.ambient * material.ambient;
    diffuse = light.diffuse * diff * material.diffuse;
    vec3 reflectDir = reflect(-lightDir, normal);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), material.shininess);
    specular = light.specular * spec * material.specular;
  }

  ambient *= attenuation;
  diffuse *= attenuation;
  specular *= attenuation;
  return (ambient / pointLightNum + diffuse + specular);
}