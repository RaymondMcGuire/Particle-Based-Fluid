/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:50:18 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:50:18 
 */
#include <kiri_core/material/material_ssao_blur.h>

void KiriMaterialSSAOBlur::Setup()
{
    KiriMaterial::Setup();
    mShader->Use();
    mShader->SetInt("ssaoInput", 0);
}

void KiriMaterialSSAOBlur::Update()
{
    mShader->Use();
}

KiriMaterialSSAOBlur::KiriMaterialSSAOBlur()
{
    mName = "ssao_blur";
    Setup();
}