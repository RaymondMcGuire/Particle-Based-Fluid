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

out VS_OUT {
  vec3 FragPos;
  vec2 TexCoords;
  vec3 Normal;
  mat3 TBN;
}
vs_out;

uniform mat4 model;
uniform bool invert;

void main() {
  vec3 normal = invert ? -aNormal : aNormal;
  vec4 worldPos = model * vec4(aPos, 1.0);
  vec4 viewPos = view * worldPos;
  vs_out.FragPos = viewPos.xyz;
  vs_out.TexCoords = aTexCoords;

  mat3 normalMatrix = transpose(inverse(mat3(view * model)));
  vs_out.Normal = normalMatrix * normal;

  // TBN
  vec3 T = normalize(vec3(view * model * vec4(aTangent, 0.0)));
  vec3 N = normalize(vec3(view * model * vec4(normal, 0.0)));
  T = normalize(T - dot(T, N) * N);
  vec3 B = cross(N, T);
  vs_out.TBN = mat3(T, B, N);

  gl_Position = projection * view * worldPos;
}