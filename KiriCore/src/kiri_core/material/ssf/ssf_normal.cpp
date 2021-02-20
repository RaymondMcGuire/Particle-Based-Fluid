/*** 
 * @Author: Xu.WANG
 * @Date: 2021-01-20 13:06:37
 * @LastEditTime: 2021-02-19 18:09:12
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\material\ssf\ssf_normal.cpp
 */

#include <kiri_core/material/ssf/ssf_normal.h>
#include <kiri_application.h>
void KiriMaterialSSFNormal::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
}

void KiriMaterialSSFNormal::Update()
{
    auto &app = KIRI::KiriApplication::Get();
    UInt height = app.GetWindow().GetWindowHeight();
    UInt width = app.GetWindow().GetWindowWidth();

    mShader->Use();

    mShader->SetFloat("particleScale", mParticleScale);
    mShader->SetFloat("screenWidth", (float)width);
    mShader->SetFloat("screenHeight", (float)height);

    mShader->SetBool("keepEdge", 1);
}

void KiriMaterialSSFNormal::SetDepthTex(Int id)
{
    mShader->SetInt("depthTex", id);
}

KiriMaterialSSFNormal::KiriMaterialSSFNormal()
{
    mName = "ssf_normal";
    Setup();
}

void KiriMaterialSSFNormal::SetParticleScale(float particleScale)
{
    mParticleScale = particleScale;
}