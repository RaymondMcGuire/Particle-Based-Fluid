/***
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:13:56
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \KiriCore\include\kiri_core\material\material_pbr_ibl.h
 */

#ifndef _KIRI_MATERIAL_PBR_IBL_H_
#define _KIRI_MATERIAL_PBR_IBL_H_
#pragma once
#include <kiri_core/light/point_light.h>
#include <kiri_core/material/material.h>


class KiriMaterialPBRIBL : public KiriMaterial {
public:
  KiriMaterialPBRIBL(UInt, UInt, UInt);

  void Setup() override;
  void Update() override;

  void SetPointLights(Array1<KiriPointLightPtr> _pointLights) {
    pointLights = _pointLights;
  }

  void SetAlbedo(Vector3F _albedo) { mAlbedo = _albedo; }

  void SetMetallic(float _metallic) { metallic = _metallic; }

  void SetRoughness(float _roughness) { roughness = _roughness; }

  void SetAo(float _ao) { ao = _ao; }

private:
  Vector3F mAlbedo;
  float metallic, roughness, ao;
  Array1<KiriPointLightPtr> pointLights;

  UInt irradianceMap;
  UInt spec_cubemap;
  UInt brdfLUT;
};

typedef SharedPtr<KiriMaterialPBRIBL> KiriMaterialPBRIBLPtr;
#endif