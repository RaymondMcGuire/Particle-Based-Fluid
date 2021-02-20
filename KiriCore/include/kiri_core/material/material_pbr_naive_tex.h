/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:13:44
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_pbr_naive_tex.h
 */

#ifndef _KIRI_MATERIAL_PBR_NAIVE_TEX_H_
#define _KIRI_MATERIAL_PBR_NAIVE_TEX_H_
#pragma once
#include <kiri_core/material/material.h>
#include <kiri_core/light/point_light.h>

class KiriMaterialPBRNaiveTex : public KiriMaterial
{
public:
    KiriMaterialPBRNaiveTex();

    void Setup() override;
    void Update() override;

    void SetPointLights(Array1<KiriPointLight> _pointLights)
    {
        pointLights = _pointLights;
    }

    void SetAlbedo(UInt _albedo)
    {
        mAlbedo = _albedo;
    }

    void SetNormal(UInt _normal)
    {
        normal = _normal;
    }

    void SetMetallic(UInt _metallic)
    {
        metallic = _metallic;
    }

    void SetRoughness(UInt _roughness)
    {
        roughness = _roughness;
    }

    void SetAo(UInt _ao)
    {
        ao = _ao;
    }

private:
    UInt mAlbedo, normal, metallic, roughness, ao;
    Array1<KiriPointLight> pointLights;
};

typedef SharedPtr<KiriMaterialPBRNaiveTex> KiriMaterialPBRNaiveTexPtr;
#endif