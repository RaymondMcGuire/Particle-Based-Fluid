/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:49:34 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:49:34 
 */
#include <kiri_core/material/material_pbr_naive_tex.h>

void KiriMaterialPBRNaiveTex::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
    mShader->Use();
    mShader->SetInt("albedoMap", 0);
    mShader->SetInt("normalMap", 1);
    mShader->SetInt("metallicMap", 2);
    mShader->SetInt("roughnessMap", 3);
    mShader->SetInt("aoMap", 4);
}

void KiriMaterialPBRNaiveTex::Update()
{
    mShader->Use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mAlbedo);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, normal);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, metallic);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, roughness);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, ao);

    for (UInt i = 0; i < pointLights.size(); ++i)
    {
        mShader->SetVec3("lightPositions[" + std::to_string(i) + "]", pointLights[i].position);
        mShader->SetVec3("lightColors[" + std::to_string(i) + "]", pointLights[i].diffuse);
    }
}

KiriMaterialPBRNaiveTex::KiriMaterialPBRNaiveTex()
{
    mName = "pbr_naive_tex";
    Setup();
}
