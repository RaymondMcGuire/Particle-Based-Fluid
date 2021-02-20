/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 18:38:22
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\material\ssf\ssf_thick.cpp
 */

#include <kiri_core/material/ssf/ssf_thick.h>

void KiriMaterialSSFThick::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
}

void KiriMaterialSSFThick::Update()
{
    mShader->Use();

    mShader->SetFloat("particleScale", mParticleScale);
    mShader->SetFloat("particleSize", mParticleRadius);
}

KiriMaterialSSFThick::KiriMaterialSSFThick()
{
    mName = "ssf_thick";
    Setup();
}

void KiriMaterialSSFThick::SetParticleScale(float particleScale)
{
    mParticleScale = particleScale;
}

void KiriMaterialSSFThick::SetParticleRadius(float particleRadius)
{
    mParticleRadius = particleRadius;
}