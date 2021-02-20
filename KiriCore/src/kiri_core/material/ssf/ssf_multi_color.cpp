/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2020-11-02 21:15:23
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\material\ssf\ssf_multi_color.cpp
 */

#include <kiri_core/material/ssf/ssf_multi_color.h>

void KiriMaterialSSFMultiColor::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
}

void KiriMaterialSSFMultiColor::Update()
{
    mShader->Use();
    mShader->SetFloat("particleScale", mParticleScale);
    mShader->SetFloat("particleSize", mParticleRadius);
    mShader->SetBool("transparent", bFluidTransparent);
}

KiriMaterialSSFMultiColor::KiriMaterialSSFMultiColor()
{
    mName = "ssf_multi_color";
    Setup();
}

void KiriMaterialSSFMultiColor::SetParticleScale(float particleScale)
{
    mParticleScale = particleScale;
}

void KiriMaterialSSFMultiColor::SetParticleRadius(float particleRadius)
{
    mParticleRadius = particleRadius;
}

void KiriMaterialSSFMultiColor::SetTransparentMode(bool bFluidTransparent)
{
    bFluidTransparent = bFluidTransparent;
}