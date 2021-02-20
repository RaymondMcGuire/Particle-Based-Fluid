/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:48:37 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:48:37 
 */
#include <kiri_core/material/material_cube_skbr.h>

void KiriMaterialCubeSKBR::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();

    mShader->Use();
    mShader->SetInt("skybox", 0);
}

void KiriMaterialCubeSKBR::Update()
{

    mShader->Use();
    mShader->SetBool("reflection", reflection);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubeSkyboxTexture);
}

KiriMaterialCubeSKBR::KiriMaterialCubeSKBR()
{
    mName = "cube_skbr";
    Setup();
}

KiriMaterialCubeSKBR::KiriMaterialCubeSKBR(UInt _cubeSkyboxTexture, bool _reflection)
{
    mName = "cube_skbr";
    reflection = _reflection;
    cubeSkyboxTexture = _cubeSkyboxTexture;
    Setup();
}
