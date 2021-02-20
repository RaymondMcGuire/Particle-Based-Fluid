/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:48:59 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:48:59 
 */
#include <kiri_core/material/material_hdr.h>

void KiriMaterialHDR::Setup()
{
    KiriMaterial::Setup();
    mShader->Use();
    mShader->SetInt("bloom", bloom);
    mShader->SetInt("sceneTex", 0);
    mShader->SetInt("bloomTex", 1);
}

void KiriMaterialHDR::SetSceneBuffer(UInt _sceneBuffer)
{
    sceneBuffer = _sceneBuffer;
}

void KiriMaterialHDR::SetBloomBuffer(UInt _bloomBuffer)
{
    bloomBuffer = _bloomBuffer;
}

void KiriMaterialHDR::SetExposure(float _exposure)
{
    exposure = _exposure;
}

void KiriMaterialHDR::Update()
{
    mShader->Use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, sceneBuffer);
    if (bloom)
    {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, bloomBuffer);
    }
    mShader->SetInt("hdr", hdr);
    mShader->SetFloat("exposure", exposure);
}

void KiriMaterialHDR::SetBloom(bool _bloom)
{
    bloom = _bloom;
    mShader->SetInt("bloom", bloom);
}

void KiriMaterialHDR::SetHDR(bool _hdr)
{
    hdr = _hdr;
}

KiriMaterialHDR::KiriMaterialHDR(bool _bloom)
{
    mName = "hdr";
    bloom = _bloom;
    hdr = true;
    Setup();
}