/*** 
 * @Author: Xu.WANG
 * @Date: 2020-06-16 01:32:28
 * @LastEditTime: 2020-11-12 19:39:47
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\material\ssf\ssf_smooth.cpp
 */

#include <kiri_core/material/ssf/ssf_smooth.h>

void KiriMaterialSSFSmooth::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
}

void KiriMaterialSSFSmooth::Update()
{
    mShader->Use();

    mShader->SetInt("kernelR", mkernelR);
    mShader->SetFloat("sigmaR", mSigmaR);
    mShader->SetFloat("sigmaZ", mSigmaZ);

    mShader->SetInt("blurOption", 0);
}

void KiriMaterialSSFSmooth::SetSmoothTex(Int id)
{
    mShader->SetInt("zA", id);
}

void KiriMaterialSSFSmooth::SetRealDepthTex(Int id)
{
    mShader->SetInt("zB", id);
}

void KiriMaterialSSFSmooth::SetMultiColorTex(Int id)
{
    mShader->SetInt("multiColorTex", id);
}

void KiriMaterialSSFSmooth::SetParticleView(bool enable_particle_view)
{
    mShader->SetBool("particleView", enable_particle_view);
}

void KiriMaterialSSFSmooth::SetEnableSSF(bool enable_ssf)
{
    mShader->SetInt("enableSSF", enable_ssf);
}

KiriMaterialSSFSmooth::KiriMaterialSSFSmooth()
{
    mName = "ssf_smooth";
    mkernelR = 10;
    mSigmaR = 1 / 6.f;
    mSigmaZ = 1 / 0.1f;
    Setup();
}