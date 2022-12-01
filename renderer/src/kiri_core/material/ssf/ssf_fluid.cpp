/*** 
 * @Author: Xu.WANG
 * @Date: 2021-02-19 22:14:23
 * @LastEditTime: 2021-02-20 02:10:50
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\material\ssf\ssf_fluid.cpp
 */

#include <kiri_core/material/ssf/ssf_fluid.h>

void KiriMaterialSSFFluid::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
}

void KiriMaterialSSFFluid::SetCameraParams(float aspect, float zFar, float zNear, float zFov, Matrix4x4F invMat)
{
    mAspect = aspect;
    mFar = zFar;
    mNear = zNear;
    mFov = zFov;
    mInvMat = invMat;
}

void KiriMaterialSSFFluid::Update()
{
    mShader->Use();
    mShader->SetFloat("r0", mR0);
    mShader->SetFloat("aspect", mAspect);
    mShader->SetFloat("far", mFar);
    mShader->SetFloat("near", mNear);

    float tanfFov = tanf(kiri_math::degreesToRadians<float>(mFov) * 0.5f);

    mShader->SetFloat("tanfFov", tanfFov);

    mShader->SetMat4("inverseView", mInvMat);

    // effect
    mShader->SetVec3("dirLight.direction", mDefaultDirectLight.direction);
    mShader->SetVec3("dirLight.ambient", mDefaultDirectLight.ambient);
    mShader->SetVec3("dirLight.diffuse", mDefaultDirectLight.diffuse);
    mShader->SetVec3("dirLight.specular", mDefaultDirectLight.specular);

    mShader->SetVec4("liquidColor", 0.275f, 0.65f, 0.85f, 0.5f);

    mShader->SetBool("multiColor", bMultiColor);
    mShader->SetInt("renderOpt", mRenderOpt);
}

void KiriMaterialSSFFluid::SetParticleView(bool enable_particle_view)
{
    mShader->SetBool("particleView", enable_particle_view);
}

void KiriMaterialSSFFluid::SetDepthTex(Int id)
{
    mShader->SetInt("depthTex", id);
}

void KiriMaterialSSFFluid::SetNormalTex(Int id)
{
    mShader->SetInt("normalTex", id);
}

void KiriMaterialSSFFluid::SetThickTex(Int id)
{
    mShader->SetInt("thickTex", id);
}

void KiriMaterialSSFFluid::SetMultiColorTex(Int id)
{
    mShader->SetInt("multiColorTex", id);
}

void KiriMaterialSSFFluid::SetBgDepthTex(Int id)
{
    mShader->SetInt("bgDepthTex", id);
}

void KiriMaterialSSFFluid::SetSkyBoxTex(Int id)
{
    mShader->SetInt("skyBoxTex", id);
}

void KiriMaterialSSFFluid::SetBGTex(Int id)
{
    mShader->SetInt("backgroundTex", id);
}

void KiriMaterialSSFFluid::SetMultiColor(bool multiColor)
{
    bMultiColor = multiColor;
}

void KiriMaterialSSFFluid::SetRenderOpt(Int renderOpt)
{
    mRenderOpt = renderOpt;
}

KiriMaterialSSFFluid::KiriMaterialSSFFluid()
{
    mName = "ssf_fluid";

    float n1 = 1.3333f;
    float t = (n1 - 1) / (n1 + 1);
    mR0 = t * t;

    bMultiColor = false;
    mRenderOpt = 4;
    Setup();
}