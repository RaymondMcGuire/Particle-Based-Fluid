/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:12:13
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_pbr_naive.h
 */

#ifndef _KIRI_MATERIAL_PBR_NAIVE_H_
#define _KIRI_MATERIAL_PBR_NAIVE_H_
#pragma once

#include <kiri_core/material/material.h>
#include <kiri_core/light/point_light.h>

class KiriMaterialPBRNaive : public KiriMaterial
{
public:
    KiriMaterialPBRNaive();

    void Setup() override;
    void Update() override;

    void SetPointLights(Array1<KiriPointLight> _pointLights)
    {
        pointLights = _pointLights;
    }

    void SetAlbedo(Vector3F _albedo)
    {
        mAlbedo = _albedo;
    }

    void SetMetallic(float _metallic)
    {
        metallic = _metallic;
    }

    void SetRoughness(float _roughness)
    {
        roughness = _roughness;
    }

    void SetAo(float _ao)
    {
        ao = _ao;
    }

private:
    Vector3F mAlbedo;
    float metallic, roughness, ao;
    Array1<KiriPointLight> pointLights;
};

typedef SharedPtr<KiriMaterialPBRNaive> KiriMaterialPBRNaivePtr;
#endif