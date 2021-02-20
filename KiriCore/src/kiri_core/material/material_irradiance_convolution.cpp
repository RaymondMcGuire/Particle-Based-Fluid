/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:49:19 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:49:19 
 */
#include <kiri_core/material/material_irradiance_convolution.h>

void KiriMaterialIrradianceConvolution::Setup()
{
    KiriMaterial::Setup();
    mShader->Use();
    mShader->SetInt("environmentMap", 0);
    mShader->SetMat4("projection", mCaptureProjection);
}

void KiriMaterialIrradianceConvolution::Update()
{
    mShader->Use();
}

KiriMaterialIrradianceConvolution::KiriMaterialIrradianceConvolution(Matrix4x4F _captureProjection)
{
    mName = "irradiance_convolution";
    mCaptureProjection = _captureProjection;
    Setup();
}