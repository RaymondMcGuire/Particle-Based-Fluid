#version 330 core

in VS_OUT { vec2 texCoord; }
fs_in;

layout(std140) uniform Matrices {
  mat4 projection;
  mat4 view;
  vec3 camPos;
};

struct DirLight {
  vec3 direction;
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
};

uniform DirLight dirLight;
uniform mat4 inverseView;

// ssf params
uniform bool multiColor;
uniform bool particleView;
uniform vec4 liquidColor;

uniform int renderOpt;

uniform float r0;

// camera params
uniform float aspect;
uniform float tanfFov;
uniform float far;
uniform float near;

// texture
uniform sampler2D depthTex;
uniform sampler2D normalTex;
uniform sampler2D thickTex;
uniform sampler2D multiColorTex;
uniform sampler2D bgDepthTex;

uniform samplerCube skyBoxTex;
uniform sampler2D backgroundTex;

out vec4 FragColor;
out vec4 brightColor;

float proj(float ze) {
  return (far + near) / (far - near) + 2 * far * near / ((far - near) * ze);
}

vec3 uv2Eye() {
  float z = texture(depthTex, fs_in.texCoord).x;
  float x = fs_in.texCoord.x, y = fs_in.texCoord.y;
  x = (2 * x - 1) * aspect * tanfFov * z;
  y = (2 * y - 1) * tanfFov * z;
  return vec3(x, y, -z);
}

void render_depth() {
  float z = texture(depthTex, fs_in.texCoord).x;
  float c = exp(z) / (exp(z) + 1);
  c = (c - 0.5) * 2;

  FragColor = vec4(c, c, c, 1);
}

void render_thick() {
  vec3 n = texture(normalTex, fs_in.texCoord).xyz;
  vec3 p = uv2Eye();
  vec3 e = normalize(-p);

  float t = texture(thickTex, fs_in.texCoord).x;

  t = exp(t) / (exp(t) + 1);
  t = (t - 0.5) * 2;

  FragColor = vec4(t, t, t, 1);
}

void render_normal() {
  FragColor = vec4(texture(normalTex, fs_in.texCoord).xyz, 1.0);
}

void render_color() {
  if (multiColor) {
    FragColor = vec4(texture(multiColorTex, fs_in.texCoord).xyz, 1.0);
  } else {
    FragColor = liquidColor;
  }
}

void render_particle() {
  float w = texture(multiColorTex, fs_in.texCoord).w;

  vec4 currentLiquidColor = liquidColor;
  if (multiColor) {
    currentLiquidColor = vec4(texture(multiColorTex, fs_in.texCoord).xyz, 1.0);
  }
  vec3 n = texture(normalTex, fs_in.texCoord).xyz;
  vec3 p = uv2Eye();
  // blinn-phong
  vec3 viewDir = -normalize(p);
  vec3 lightDir = normalize((view * vec4(dirLight.direction, 0.0f)).xyz);
  vec3 halfVec = normalize(viewDir + lightDir);
  vec3 specular =
      vec3(dirLight.specular * pow(max(dot(halfVec, n), 0.0f), 400.0f));
  vec3 diffuse = currentLiquidColor.xyz * max(dot(lightDir, n), 0.0f) *
                 dirLight.diffuse * currentLiquidColor.w;

  FragColor.rgb = diffuse + specular + currentLiquidColor.rgb * 0.5f;

  if (w == 0.0)
    FragColor.a = 0.f;
  else
    FragColor.a = currentLiquidColor.a;

  float brightness = dot(FragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
  brightColor = vec4(FragColor.rgb * brightness * brightness, 1.0f);
}

vec3 trace_refract_color(vec3 refract_dir) {
  float refractScale = 0.15;
  return texture(backgroundTex, fs_in.texCoord - refract_dir.xy * refractScale)
      .rgb;
}

vec3 trace_reflect_color(vec3 reflect_dir) {
  return texture(skyBoxTex, reflect_dir).rgb;
}

void render_fluid() {

  vec3 n = texture(normalTex, fs_in.texCoord).xyz;
  vec3 p = uv2Eye();
  vec3 e = normalize(-p);
  float r = r0 + (1 - r0) * pow(1 - dot(n, e), 3);

  vec3 reflect_dir = reflect(-e, n);

  float ratio = 1.00 / 1.52;
  vec3 refract_dir = refract(-e, n, ratio);

  float thickness = texture(thickTex, fs_in.texCoord).x;
  float attenuate = max(exp(0.5 * -thickness), 0.2);

  vec4 currentLiquidColor = liquidColor;
  if (multiColor) {
    currentLiquidColor = vec4(texture(multiColorTex, fs_in.texCoord).xyz, 1.0);
  }

  vec3 refract_color =
      mix(currentLiquidColor.rgb, trace_refract_color(refract_dir), attenuate);
  vec3 reflect_color = trace_reflect_color(reflect_dir);

  float w = texture(multiColorTex, fs_in.texCoord).w;
  // solid
  vec4 combine_color = vec4(currentLiquidColor.xyz * 0.5, 1);

  // liquid
  if (w == 0.0 && !particleView) {
    // I_reflect*r + I_refract*(1-r)
    combine_color = vec4(mix(refract_color, reflect_color, r), 1.f);
  } else if (particleView) {
    // particle
    combine_color = vec4(currentLiquidColor.xyz * 0.5, currentLiquidColor.w);
  }

  // blinn-phong
  vec3 viewDir = -normalize(p);
  vec3 lightDir = normalize((view * vec4(dirLight.direction, 0.0f)).xyz);
  vec3 halfVec = normalize(viewDir + lightDir);
  vec3 specular =
      vec3(dirLight.specular * pow(max(dot(halfVec, n), 0.0f), 400.0f));
  vec3 diffuse = currentLiquidColor.xyz * max(dot(lightDir, n), 0.0f) *
                 dirLight.diffuse * currentLiquidColor.w;

  FragColor.rgb = diffuse + specular + combine_color.rgb;
  FragColor.a = combine_color.a;

  float brightness = dot(FragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
  brightColor = vec4(FragColor.rgb * brightness * brightness, 1.0f);
}

float LinearizeDepth(float depth) {
  float z = depth * 2.0 - 1.0; // back to NDC
  return (2.0 * near * far) / (far + near - z * (far - near));
}

void main() {

  // ze to z_ndc to gl_FragDepth
  // REF:
  // https://computergraphics.stackexchange.com/questions/6308/why-does-this-gl-fragdepth-calculation-work?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
  float ze = texture(depthTex, fs_in.texCoord).x;
  float z_ndc = proj(-ze);
  float ssfz = 0.5 * (gl_DepthRange.diff * z_ndc + gl_DepthRange.far +
                      gl_DepthRange.near);

  float bz = texture(bgDepthTex, fs_in.texCoord).x;

  // ze <=LinearizeDepth(bz)
  if (ssfz <= bz && ze <= 50) {
    if (renderOpt == 0) {
      render_depth();
    } else if (renderOpt == 1) {
      render_thick();
    } else if (renderOpt == 2) {
      render_normal();
    } else if (renderOpt == 3) {
      render_color();
    } else if (renderOpt == 4) {
      render_fluid();
    } else {
      render_fluid();
    }

  } else {
    FragColor = texture(backgroundTex, fs_in.texCoord);
  }
}