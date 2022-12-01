/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:48:43 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:48:43 
 */
#include <kiri_core/material/material_debug_normal.h>

void KiriMaterialDebugNomral::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
    mShader->Use();
}

void KiriMaterialDebugNomral::Update()
{
    mShader->Use();
}

KiriMaterialDebugNomral::KiriMaterialDebugNomral()
{
    mName = "normal_visualization";
    KiriMaterial::GeoShaderEnable();
    Setup();
}