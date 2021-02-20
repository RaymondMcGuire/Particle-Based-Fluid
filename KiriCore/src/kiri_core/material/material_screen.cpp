/*
 * @Author: Xu.Wang 
 * @Date: 2020-05-14 22:31:43 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-14 23:00:47
 */

#include <kiri_core/material/material_screen.h>

void KiriMaterialScreen::Setup()
{
    KiriMaterial::Setup();
}

void KiriMaterialScreen::SetPostProcessingType(Int type)
{
    mPostProcessingType = type;
}

void KiriMaterialScreen::Update()
{
    mShader->Use();
    mShader->SetInt("screenTexture", 0);
    mShader->SetInt("post_processing_type", mPostProcessingType);
}

KiriMaterialScreen::KiriMaterialScreen()
{
    mName = "screen";
    mPostProcessingType = 0;
    Setup();
}
