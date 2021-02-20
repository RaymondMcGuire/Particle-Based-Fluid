/*** 
 * @Author: Xu.WANG
 * @Date: 2020-06-16 01:32:28
 * @LastEditTime: 2021-02-20 18:41:22
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\src\kiri_core\material\ssf\ssf_depth.cpp
 */

#include <kiri_core/material/ssf/ssf_depth.h>

void KiriMaterialSSFDepth::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
}

void KiriMaterialSSFDepth::Update()
{
    mShader->Use();

    mShader->SetFloat("particleScale", mParticleScale);
    mShader->SetFloat("particleSize", mParticleRadius);
}

KiriMaterialSSFDepth::KiriMaterialSSFDepth()
{
    mName = "ssf_depth";
    Setup();
}

void KiriMaterialSSFDepth::SetParticleScale(float particleScale)
{
    mParticleScale = particleScale;
}

void KiriMaterialSSFDepth::SetParticleRadius(float particleRadius)
{
    mParticleRadius = particleRadius;
}