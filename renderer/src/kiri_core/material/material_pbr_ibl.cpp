/***
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-12-23 14:02:35
 * @LastEditors: Xu.WANG
 * @Description:
 */
#include <kiri_core/material/material_pbr_ibl.h>

void KiriMaterialPBRIBL::Setup() {
  KiriMaterial::Setup();
  BindGlobalUniformBufferObjects();
  mShader->Use();
  mShader->SetInt("irradianceMap", 0);
  mShader->SetInt("spec_cubemap", 1);
  mShader->SetInt("brdfLUT", 2);
}

void KiriMaterialPBRIBL::Update() {
  mShader->Use();
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_CUBE_MAP, spec_cubemap);
  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, brdfLUT);

  mShader->SetVec3("albedo", mAlbedo);
  mShader->SetFloat("metallic", metallic);
  mShader->SetFloat("roughness", roughness);
  mShader->SetFloat("ao", ao);
  for (UInt i = 0; i < pointLights.size(); ++i) {
    mShader->SetVec3("lightPositions[" + std::to_string(i) + "]",
                     pointLights[i]->position);
    mShader->SetVec3("lightColors[" + std::to_string(i) + "]",
                     pointLights[i]->diffuse);
  }
}

KiriMaterialPBRIBL::KiriMaterialPBRIBL(UInt _irradianceMap, UInt _specCubeMap,
                                       UInt _brdfLUT) {
  mName = "pbr_ibl";
  irradianceMap = _irradianceMap;
  spec_cubemap = _specCubeMap;
  brdfLUT = _brdfLUT;
  mAlbedo = Vector3F(0.5f, 0.0f, 0.0f);
  metallic = 0.1f;
  roughness = 0.1f;
  ao = 1.0f;
  Setup();
}
