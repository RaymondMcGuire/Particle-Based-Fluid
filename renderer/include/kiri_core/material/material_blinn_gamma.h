/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:17:08
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_blinn_gamma.h
 */

#ifndef _KIRI_MATERIAL_BLINN_GAMMA_H_
#define _KIRI_MATERIAL_BLINN_GAMMA_H_
#pragma once
#include <kiri_core/material/material.h>
#include <kiri_core/light/point_light.h>

class KiriMaterialBlinnGamma : public KiriMaterial
{
public:
    KiriMaterialBlinnGamma(UInt, UInt);

    void Setup() override;
    void Update() override;

    void SetGamma(bool);
    void SetPointLights(Array1<KiriPointLightPtr>);

private:
    UInt gammaTex;
    UInt tex;
    bool gamma;

    Array1<KiriPointLightPtr> pointLights;
};
typedef SharedPtr<KiriMaterialBlinnGamma> KiriMaterialBlinnGammaPtr;
#endif