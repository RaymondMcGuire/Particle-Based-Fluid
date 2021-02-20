/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:12:41
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\src\kiri_core\material\material_pbr_naive.cpp
 */

#include <kiri_core/material/material_pbr_naive.h>

void KiriMaterialPBRNaive::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
    mShader->Use();
}

void KiriMaterialPBRNaive::Update()
{
    mShader->Use();

    mShader->SetVec3("albedo", mAlbedo);
    mShader->SetFloat("metallic", metallic);
    mShader->SetFloat("roughness", roughness);
    mShader->SetFloat("ao", ao);
    for (UInt i = 0; i < pointLights.size(); ++i)
    {
        mShader->SetVec3("lightPositions[" + std::to_string(i) + "]", pointLights[i].position);
        mShader->SetVec3("lightColors[" + std::to_string(i) + "]", pointLights[i].diffuse);
    }
}

KiriMaterialPBRNaive::KiriMaterialPBRNaive()
{
    mName = "pbr_naive";
    mAlbedo = Vector3F(0.5f, 0.0f, 0.0f);
    metallic = 0.1f;
    roughness = 0.1f;
    ao = 1.0f;
    Setup();
}
