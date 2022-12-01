/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:48:40 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:48:40 
 */
#include <kiri_core/material/material_cube_skybox.h>

void KiriMaterialCubeSkybox::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
    mShader->Use();
    mShader->SetInt("skybox", 0);
}

void KiriMaterialCubeSkybox::Update()
{
    mShader->Use();
}

KiriMaterialCubeSkybox::KiriMaterialCubeSkybox()
{
    mName = "cube_skybox";
    Setup();
}