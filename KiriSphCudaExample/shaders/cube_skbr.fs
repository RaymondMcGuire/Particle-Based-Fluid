#version 330 core
out vec4 FragColor;

layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
};

in vec3 Normal;
in vec3 Position;

uniform samplerCube skybox;
uniform bool reflection;

void main() {
  vec3 I = normalize(Position - camPos);

  vec3 R;
  if (reflection) {
    R = reflect(I, normalize(Normal));
  } else {
    float ratio = 1.00 / 1.52;
    vec3 I = normalize(Position - camPos);
    R = refract(I, normalize(Normal), ratio);
  }

  FragColor = vec4(texture(skybox, R).rgb, 1.0);
}