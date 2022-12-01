/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:48:23 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:48:23 
 */
#include <kiri_core/material/material_blinn_gamma.h>

void KiriMaterialBlinnGamma::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();

    mShader->Use();
    mShader->SetInt("material.diffuse", 0);
    mShader->SetFloat("material.shininess", 64.0f);
}

void KiriMaterialBlinnGamma::Update()
{
    mShader->Use();

    //point lights
    for (size_t i = 0; i < pointLights.size(); i++)
    {
        mShader->SetVec3("pointLights[" + std::to_string(i) + "].position", pointLights[i]->position);
        mShader->SetFloat("pointLights[" + std::to_string(i) + "].constant", pointLights[i]->constant);
        mShader->SetFloat("pointLights[" + std::to_string(i) + "].linear", pointLights[i]->linear);
        mShader->SetFloat("pointLights[" + std::to_string(i) + "].quadratic", pointLights[i]->quadratic);
        mShader->SetVec3("pointLights[" + std::to_string(i) + "].ambient", pointLights[i]->ambient);
        mShader->SetVec3("pointLights[" + std::to_string(i) + "].diffuse", pointLights[i]->diffuse);
        mShader->SetVec3("pointLights[" + std::to_string(i) + "].specular", pointLights[i]->specular);
    }

    mShader->SetInt("pointLightNum", (Int)pointLights.size());
    mShader->SetInt("gamma", gamma);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gamma ? gammaTex : tex);
}

void KiriMaterialBlinnGamma::SetGamma(bool _gamma)
{
    gamma = _gamma;
}

void KiriMaterialBlinnGamma::SetPointLights(Array1<KiriPointLightPtr> _pointLights)
{
    pointLights = _pointLights;
}

KiriMaterialBlinnGamma::KiriMaterialBlinnGamma(UInt _tex, UInt _gammaTex)
{
    mName = "blinn_gamma";
    gamma = false;
    tex = _tex;
    gammaTex = _gammaTex;
    Setup();
}
