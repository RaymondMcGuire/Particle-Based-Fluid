/*
 * @Author: Xu.Wang 
 * @Date: 2020-05-09 20:34:29 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-10 00:47:35
 */

#include <kiri_core/material/particle/particle_point_sprite.h>

void KiriMaterialParticlePointSprite::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
}

void KiriMaterialParticlePointSprite::Update()
{
    mShader->Use();

    mShader->SetVec3("baseColor", mBaseColor);
    mShader->SetFloat("particleScale", mParticleScale);
    mShader->SetFloat("particleSize", mParticleRadius);

    mShader->SetVec3("dirLight.direction", mDefaultDirectLight.direction);
    mShader->SetVec3("dirLight.ambient", mDefaultDirectLight.ambient);
    mShader->SetVec3("dirLight.diffuse", mDefaultDirectLight.diffuse);
    mShader->SetVec3("dirLight.specular", mDefaultDirectLight.specular);
}

KiriMaterialParticlePointSprite::KiriMaterialParticlePointSprite()
{
    mName = "particle_point_sprite";
    mParticleScale = 1.0f;
    Setup();
}

void KiriMaterialParticlePointSprite::SetParticleScale(float particleScale)
{
    mParticleScale = particleScale;
}

void KiriMaterialParticlePointSprite::SetParticleRadius(float particleRadius)
{
    mParticleRadius = particleRadius;
}
