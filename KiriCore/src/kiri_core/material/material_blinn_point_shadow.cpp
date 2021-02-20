/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:48:26 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:48:26 
 */
#include <kiri_core/material/material_blinn_point_shadow.h>

void KiriMaterialBlinnPointShadow::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
    mShader->Use();
    mShader->SetInt("diffuseTexture", 0);
    mShader->SetInt("depthMap", 1);
}

void KiriMaterialBlinnPointShadow::Update()
{

    mShader->Use();

    mShader->SetVec3("mLightPos", Vector3F(shadow->pointLight.x, shadow->pointLight.y, shadow->pointLight.z));
    mShader->SetInt("shadows", 1);
    mShader->SetFloat("mFarPlane", shadow->mFarPlane);

    if (outside)
    {
        mShader->SetInt("reverse_normals", 0);
    }
    else
    {
        mShader->SetInt("reverse_normals", 1);
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_CUBE_MAP, shadow->getDepthCubeMap());
}

KiriMaterialBlinnPointShadow::KiriMaterialBlinnPointShadow(bool _outside, KiriPointShadow *_shadow, KiriTexture _texture)
{
    mName = "blinn_point_shadow";
    texture = _texture.Load();
    shadow = _shadow;
    outside = _outside;
    Setup();
}
