/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:16:18
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_blinn_standard.h
 */

#ifndef _KIRI_MATERIAL_BLINN_STANDARD_H_
#define _KIRI_MATERIAL_BLINN_STANDARD_H_
#pragma once
#include <kiri_core/material/material.h>
#include <kiri_core/material/material_constants.h>
#include <kiri_core/light/point_light.h>

class KiriMaterialBlinnStandard : public KiriMaterial
{
public:
    KiriMaterialBlinnStandard(bool = true, bool = false, bool = false, bool = false, bool = false);

    void Setup() override;
    void Update() override;

    // Setter Method
    void SetPointLights(Array1<KiriPointLightPtr>);

    void SetTextureMap(bool);
    void SetDiffuseTex(bool);
    void SetSpecularTex(bool);
    void SetNormalMap(bool);
    void SetInverseNormal(bool);

    void SetAttenParams(float, float, float);

    void SetConsMaterial(KIRI_MATERIAL_CONSTANT_TYPE);

private:
    void updateSetting();
    void SetConstMaterial(STRUCT_MATERIAL);

    Array1<KiriPointLightPtr> pointLights;

    // use constant material
    KIRI_MATERIAL_CONSTANT_TYPE constMaterial;

    // enable texture map
    bool bTextureMap;
    bool bDiffuseTexure;
    bool bSpecularTexure;
    bool bNormalTexure;

    // invert normal
    bool inverseNormal;

    // lightings atten
    float atten_constant;
    float atten_linear;
    float atten_quadratic;

    // post-processing
    bool gamma;
};
typedef SharedPtr<KiriMaterialBlinnStandard> KiriMaterialBlinnStandardPtr;
#endif