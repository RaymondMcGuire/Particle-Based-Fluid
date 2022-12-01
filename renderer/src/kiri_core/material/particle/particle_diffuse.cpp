/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-20 02:04:35 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-04-20 02:04:35 
 */
#include <kiri_core/material/particle/particle_diffuse.h>

void KiriMaterialParticleDiffuse::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
}

void KiriMaterialParticleDiffuse::Update()
{
    mShader->Use();
    mShader->SetVec3("particle_color", particle_color);
    mShader->SetVec3("light_direction", light_direction);
}

KiriMaterialParticleDiffuse::KiriMaterialParticleDiffuse()
{
    mName = "particle_diffuse";
    particle_color = Vector3F(255.0f, 0.0f, 0.0f);
    light_direction = Vector3F(0.1f, 0.1f, 0.1f);

    Setup();
}

void KiriMaterialParticleDiffuse::SetParticleColor(Vector3F _particle_color)
{
    particle_color = _particle_color;
}

void KiriMaterialParticleDiffuse::SetLightDirection(Vector3F _light_direction)
{
    light_direction = _light_direction;
}