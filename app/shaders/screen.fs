#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenTexture;
uniform int post_processing_type;

void main() {
  vec3 col = texture(screenTexture, TexCoords).rgb;
  float average = (col.r + col.g + col.b) / 3.0;

  if (post_processing_type == 1) {
    FragColor = vec4(average, average, average, 1.0);
  } else {
    FragColor = vec4(col, 1.0);
  }

  //FragColor = vec4(texture(screenTexture, TexCoords).xxx, 1.0);
}