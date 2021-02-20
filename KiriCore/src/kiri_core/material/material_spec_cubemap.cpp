/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:50:15 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:50:15 
 */
#include <kiri_core/material/material_spec_cubemap.h>

void KiriMaterialSpecCubeMap::Setup()
{
    KiriMaterial::Setup();
    mShader->Use();
    mShader->SetInt("environmentMap", 0);
    mShader->SetMat4("projection", mCaptureProjection);
}

void KiriMaterialSpecCubeMap::Update()
{
    mShader->Use();
}

KiriMaterialSpecCubeMap::KiriMaterialSpecCubeMap(Matrix4x4F _captureProjection)
{
    mCaptureProjection = _captureProjection;
    mName = "spec_cubemap";
    Setup();
}