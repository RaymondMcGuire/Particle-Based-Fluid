#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 3) in mat4 aInstanceMatrix;

layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
};

out VS_OUT {
  vec3 FragPos;
  vec3 Normal;
  flat int Instance;
}
vs_out;

void main() {
  vs_out.FragPos = aPos;
  vs_out.Normal = aNormal;
  vs_out.Instance = gl_InstanceID;
  gl_Position = projection * view * aInstanceMatrix * vec4(aPos, 1.0f);
}