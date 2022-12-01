#version 330 core
out vec4 FragColor;

in VS_OUT {
  vec3 FragPos;
  vec3 Normal;
}
fs_in;

uniform vec3 light_direction;
uniform vec3 particle_color;

void main() {
  // ambient
  vec3 ambient = 0.05 * particle_color;
  // diffuse
  vec3 normal = normalize(fs_in.Normal);
  float diff = max(dot(light_direction, normal), 0.0);
  vec3 diffuse = diff * particle_color;

  FragColor = vec4(ambient + diffuse, 1.0);
}