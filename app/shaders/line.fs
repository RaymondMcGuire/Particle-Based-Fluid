#version 330 core
out vec4 FragColor;

in VS_OUT { vec3 color; }
fs_in;

void main()
{
    FragColor = vec4(fs_in.color.xyz, 1.0f);
} 