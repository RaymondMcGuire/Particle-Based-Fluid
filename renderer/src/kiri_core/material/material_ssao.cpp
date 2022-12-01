/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:50:21 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:50:21 
 */
#include <kiri_core/material/material_ssao.h>

void KiriMaterialSSAO::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
    mShader->Use();
    mShader->SetInt("gPosition", 0);
    mShader->SetInt("gNormal", 1);
    mShader->SetInt("texNoise", 2);
}

void KiriMaterialSSAO::Update()
{
    mShader->Use();
    for (size_t i = 0; i < mKernel.size(); ++i)
        mShader->SetVec3("samples[" + std::to_string(i) + "]", mKernel[i]);
}

KiriMaterialSSAO::KiriMaterialSSAO(Array1Vec3F _kernel)
{
    mName = "ssao";
    mKernel = _kernel;
    Setup();
}