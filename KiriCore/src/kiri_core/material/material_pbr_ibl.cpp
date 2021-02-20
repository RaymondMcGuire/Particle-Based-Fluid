/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:49:32 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:49:32 
 */
#include <kiri_core/material/material_pbr_ibl.h>

void KiriMaterialPBRIBL::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
    mShader->Use();
    mShader->SetInt("irradianceMap", 0);
    mShader->SetInt("specCubeMap", 1);
    mShader->SetInt("brdfLUT", 2);
}

void KiriMaterialPBRIBL::Update()
{
    mShader->Use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_CUBE_MAP, specCubeMap);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, brdfLUT);

    mShader->SetVec3("mAlbedo", mAlbedo);
    mShader->SetFloat("metallic", metallic);
    mShader->SetFloat("roughness", roughness);
    mShader->SetFloat("ao", ao);
    for (UInt i = 0; i < pointLights.size(); ++i)
    {
        mShader->SetVec3("lightPositions[" + std::to_string(i) + "]", pointLights[i]->position);
        mShader->SetVec3("lightColors[" + std::to_string(i) + "]", pointLights[i]->diffuse);
    }
}

KiriMaterialPBRIBL::KiriMaterialPBRIBL(UInt _irradianceMap, UInt _specCubeMap, UInt _brdfLUT)
{
    mName = "pbr_ibl";
    irradianceMap = _irradianceMap;
    specCubeMap = _specCubeMap;
    brdfLUT = _brdfLUT;
    mAlbedo = Vector3F(0.5f, 0.0f, 0.0f);
    metallic = 0.1f;
    roughness = 0.1f;
    ao = 1.0f;
    Setup();
}
