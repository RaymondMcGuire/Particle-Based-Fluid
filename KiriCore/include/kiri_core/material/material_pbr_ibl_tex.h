/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:14:05
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_pbr_ibl_tex.h
 */

#ifndef _KIRI_MATERIAL_PBR_IBL_TEX_H_
#define _KIRI_MATERIAL_PBR_IBL_TEX_H_
#pragma once
#include <kiri_core/material/material.h>
#include <kiri_core/texture/pbr_texture.h>
#include <kiri_core/light/point_light.h>

class KiriMaterialPBRIBLTex : public KiriMaterial
{
public:
    KiriMaterialPBRIBLTex(UInt, UInt, UInt, KiriPBRTexturePtr);

    void Setup() override;
    void Update() override;

    void SetPointLights(Array1<KiriPointLightPtr> _pointLights)
    {
        pointLights = _pointLights;
    }

private:
    Array1<KiriPointLightPtr> pointLights;

    UInt irradianceMap;
    UInt specCubeMap;
    UInt brdfLUT;

    UInt albedoMap;
    UInt normalMap;
    UInt metallicMap;
    UInt roughnessMap;
    UInt aoMap;
};

typedef SharedPtr<KiriMaterialPBRIBLTex> KiriMaterialPBRIBLTexPtr;
#endif