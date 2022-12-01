#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoords;
layout(location = 3) in vec3 aTangent;
layout(location = 4) in vec3 aBitangent;

layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
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

out VS_OUT {
  vec3 FragPos;
  vec2 TexCoords;
  vec3 Normal;
  mat3 TBN;
  vec3 TangentViewPos;
  vec3 TangentFragPos;
}
vs_out;

uniform bool inverse_normals;
uniform mat4 model;

void main() {
  vec3 n = inverse_normals ? -aNormal : aNormal;

  mat3 invModelT = transpose(inverse(mat3(model)));

  vs_out.FragPos = vec3(model * vec4(aPos, 1.0));
  vs_out.Normal = invModelT * n;
  vs_out.TexCoords = aTexCoords;

  vec3 T = normalize(vec3(model * vec4(aTangent, 0.0)));
  vec3 N = normalize(vec3(model * vec4(n, 0.0)));
  T = normalize(T - dot(T, N) * N);
  vec3 B = cross(N, T);
  vs_out.TBN = mat3(T, B, N);
  vs_out.TangentViewPos = vs_out.TBN * camPos;
  vs_out.TangentFragPos = vs_out.TBN * vs_out.FragPos;

  gl_Position = projection * view * vec4(vs_out.FragPos, 1.0);
}