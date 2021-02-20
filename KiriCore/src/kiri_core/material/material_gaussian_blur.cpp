/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:48:56 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:48:56 
 */
#include <kiri_core/material/material_gaussian_blur.h>

void KiriMaterialGaussianBlur::Setup()
{
    KiriMaterial::Setup();
    mShader->Use();
    mShader->SetInt("image", 0);
}

void KiriMaterialGaussianBlur::SetHorizontal(bool _h)
{
    mShader->SetInt("horizontal", _h);
}

void KiriMaterialGaussianBlur::Update()
{
    mShader->Use();
}

KiriMaterialGaussianBlur::KiriMaterialGaussianBlur(UInt _colorBuffer)
{
    mName = "gauss_blur";
    colorBuffer = _colorBuffer;
    Setup();
}