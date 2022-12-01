#version 330 core
out vec4 FragColor;

uniform vec3 particle_color;

void main() { FragColor = vec4(particle_color, 1.0); }