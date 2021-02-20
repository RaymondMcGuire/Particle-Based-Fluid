/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:50:28 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:50:28 
 */
#include <kiri_core/material/material_texture.h>

void KiriMaterialTexture::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
    mShader->Use();
}

void KiriMaterialTexture::Update()
{
    mShader->Use();
}

KiriMaterialTexture::KiriMaterialTexture()
{
    mName = "texture";
    Setup();
}