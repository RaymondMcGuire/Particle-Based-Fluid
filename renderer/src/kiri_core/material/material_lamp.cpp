/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:49:25 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-15 02:10:27
 */
#include <kiri_core/material/material_lamp.h>

void KiriMaterialLamp::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
}

void KiriMaterialLamp::Update()
{
    mShader->Use();
    mShader->SetVec3("lightColor", lightColor);
}

KiriMaterialLamp::KiriMaterialLamp(Vector3F _lightColor)
{
    mName = "lamp";
    lightColor = _lightColor;
    Setup();
}

KiriMaterialLamp::KiriMaterialLamp()
{
    mName = "lamp";
    lightColor = Vector3F(100.0f, 100.0f, 100.0f);
    Setup();
}

void KiriMaterialLamp::SetColor(Vector3F _lightColor)
{
    lightColor = _lightColor;
}