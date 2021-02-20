/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-20 02:04:37 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-04-20 02:04:37 
 */
#include <kiri_core/material/particle/particle_default.h>

void KiriMaterialParticleDefault::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
}

void KiriMaterialParticleDefault::Update()
{
    mShader->Use();
    mShader->SetVec3("particle_color", particle_color);
}

KiriMaterialParticleDefault::KiriMaterialParticleDefault()
{
    mName = "particle_default";
    Setup();
}

void KiriMaterialParticleDefault::SetParticleColor(Vector3F _particle_color)
{
    particle_color = _particle_color;
}