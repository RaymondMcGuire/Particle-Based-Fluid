/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:48:46 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-17 17:59:38
 */
#include <kiri_core/material/material_equirect2cube.h>

void KiriMaterialEquirectangular2CubeMap::Setup()
{
    KiriMaterial::Setup();
    mShader->Use();
    mShader->SetInt("equirectangularMap", 0);
    mShader->SetMat4("projection", mCaptureProjection);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, hdrTexture);
}

void KiriMaterialEquirectangular2CubeMap::Update()
{
    mShader->Use();
}

KiriMaterialEquirectangular2CubeMap::KiriMaterialEquirectangular2CubeMap(UInt _hdrTexture, Matrix4x4F _captureProjection)
{
    mName = "equirect_to_cubemap";
    hdrTexture = _hdrTexture;
    mCaptureProjection = _captureProjection;
    Setup();
}