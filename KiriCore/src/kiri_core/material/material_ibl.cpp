/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:49:04 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:49:04 
 */
#include <kiri_core/material/material_ibl.h>

void KiriMaterialIBL::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
}

void KiriMaterialIBL::Update()
{
    mShader->Use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, envMap);

    mShader->SetInt("environmentMap", 0);
}

KiriMaterialIBL::KiriMaterialIBL(UInt _envMap)
{
    mName = "ibl";
    envMap = _envMap;
    Setup();
}