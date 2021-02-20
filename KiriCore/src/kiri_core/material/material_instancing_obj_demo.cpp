/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:49:09 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:49:09 
 */
#include <kiri_core/material/material_instancing_obj_demo.h>

void KiriMaterialInstancingObjDemo::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
    mShader->Use();
}

void KiriMaterialInstancingObjDemo::Update()
{
    mShader->Use();
}

KiriMaterialInstancingObjDemo::KiriMaterialInstancingObjDemo()
{
    mName = "instancing_obj_demo";
    Setup();
}