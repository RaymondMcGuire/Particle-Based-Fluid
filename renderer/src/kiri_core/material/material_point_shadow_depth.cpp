/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:49:40 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-24 22:13:16
 */
#include <kiri_core/material/material_point_shadow_depth.h>

void KiriMaterialPointShadowDepth::SetShadowTransforms(Array1<Matrix4x4F> _shadowTransforms)
{
    mShadowTransforms = _shadowTransforms;
}

void KiriMaterialPointShadowDepth::SetFarPlane(float _far_plane)
{
    mFarPlane = _far_plane;
}

void KiriMaterialPointShadowDepth::SetLightPos(Vector3F _lightPos)
{
    mLightPos = _lightPos;
}

void KiriMaterialPointShadowDepth::Setup()
{
    KiriMaterial::Setup();
    mShader->Use();
}

void KiriMaterialPointShadowDepth::Update()
{
    mShader->Use();
    for (UInt i = 0; i < 6; ++i)
        mShader->SetMat4("shadowMatrices[" + std::to_string(i) + "]", mShadowTransforms[i]);
    mShader->SetFloat("mFarPlane", mFarPlane);
    mShader->SetVec3("mLightPos", Vector3F(mLightPos.x, mLightPos.y, mLightPos.z));
}

KiriMaterialPointShadowDepth::KiriMaterialPointShadowDepth()
{
    mName = "point_shadow_depth";
    KiriMaterial::GeoShaderEnable();
    Setup();
}
