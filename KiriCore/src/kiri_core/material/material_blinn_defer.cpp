/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:48:20 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-15 01:57:09
 */
#include <kiri_core/material/material_blinn_defer.h>

void KiriMaterialBlinnDefer::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
    mShader->Use();
    mShader->SetInt("gPosition", 0);
    mShader->SetInt("gNormal", 1);
    mShader->SetInt("gAlbedoSpec", 2);

    if (b_ssao)
        mShader->SetInt("ssao", 3);
}

void KiriMaterialBlinnDefer::Update()
{
    mShader->Use();

    for (size_t i = 0; i < pointLights.size(); i++)
    {
        mShader->SetVec3("lights[" + std::to_string(i) + "].Position", pointLights[i]->position);
        mShader->SetVec3("lights[" + std::to_string(i) + "].Color", pointLights[i]->diffuse);
        mShader->SetFloat("lights[" + std::to_string(i) + "].Linear", linear);
        mShader->SetFloat("lights[" + std::to_string(i) + "].Quadratic", quadratic);

        // calculate radius of light volume/sphere (not optimize)
        const float maxBrightness = std::fmaxf(std::fmaxf(pointLights[i]->diffuse.x, pointLights[i]->diffuse.y), pointLights[i]->diffuse.z);
        float radius = (-linear + std::sqrt(linear * linear - 4 * quadratic * (constant - (256.0f / 5.0f) * maxBrightness))) / (2.0f * quadratic);
        mShader->SetFloat("lights[" + std::to_string(i) + "].Radius", radius);
    }

    mShader->SetBool("b_ssao", b_ssao);
}

void KiriMaterialBlinnDefer::SetSSAO(bool _ssao)
{
    b_ssao = _ssao;
}

KiriMaterialBlinnDefer::KiriMaterialBlinnDefer(bool _b_ssao)
{
    mName = "blinn_defer";
    b_ssao = _b_ssao;
    Setup();
}

void KiriMaterialBlinnDefer::SetPointLights(Array1<KiriPointLightPtr> _pointLights)
{
    pointLights = _pointLights;
}