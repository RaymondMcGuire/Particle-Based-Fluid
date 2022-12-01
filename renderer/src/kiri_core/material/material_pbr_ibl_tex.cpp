/*
 * @Author: Xu.Wang
 * @Date: 2020-03-17 17:49:28
 * @Last Modified by:   Xu.Wang
 * @Last Modified time: 2020-03-17 17:49:28
 */
#include <kiri_core/material/material_pbr_ibl_tex.h>

void KiriMaterialPBRIBLTex::Setup() {
  KiriMaterial::Setup();
  BindGlobalUniformBufferObjects();
  mShader->Use();
  mShader->SetInt("irradianceMap", 0);
  mShader->SetInt("spec_cubemap", 1);
  mShader->SetInt("brdfLUT", 2);
  mShader->SetInt("albedoMap", 3);
  mShader->SetInt("normalMap", 4);
  mShader->SetInt("metallicMap", 5);
  mShader->SetInt("roughnessMap", 6);
  mShader->SetInt("aoMap", 7);
}

void KiriMaterialPBRIBLTex::Update() {
  mShader->Use();
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_CUBE_MAP, spec_cubemap);
  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, brdfLUT);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, albedoMap);
  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D, normalMap);
  glActiveTexture(GL_TEXTURE5);
  glBindTexture(GL_TEXTURE_2D, metallicMap);
  glActiveTexture(GL_TEXTURE6);
  glBindTexture(GL_TEXTURE_2D, roughnessMap);
  glActiveTexture(GL_TEXTURE7);
  glBindTexture(GL_TEXTURE_2D, aoMap);

  for (UInt i = 0; i < pointLights.size(); ++i) {
    mShader->SetVec3("lightPositions[" + std::to_string(i) + "]",
                     pointLights[i]->position);
    mShader->SetVec3("lightColors[" + std::to_string(i) + "]",
                     pointLights[i]->diffuse);
  }
}

KiriMaterialPBRIBLTex::KiriMaterialPBRIBLTex(UInt _irradianceMap,
                                             UInt _specCubeMap, UInt _brdfLUT,
                                             KiriPBRTexturePtr pbrTex) {
  mName = "pbr_ibl_tex";

  irradianceMap = _irradianceMap;
  spec_cubemap = _specCubeMap;
  brdfLUT = _brdfLUT;
  albedoMap = pbrTex->Albedo();
  normalMap = pbrTex->Normal();
  metallicMap = pbrTex->Metallic();
  roughnessMap = pbrTex->Roughness();
  aoMap = pbrTex->Ao();
  Setup();
}