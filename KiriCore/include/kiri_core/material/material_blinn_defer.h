/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:17:15
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_blinn_defer.h
 */

#ifndef _KIRI_MATERIAL_BLINN_DEFER_H_
#define _KIRI_MATERIAL_BLINN_DEFER_H_
#pragma once
#include <kiri_core/material/material.h>
#include <kiri_core/light/point_light.h>

class KiriMaterialBlinnDefer : public KiriMaterial
{
public:
    KiriMaterialBlinnDefer(bool);
    void Update() override;

    void SetPointLights(Array1<KiriPointLightPtr>);
    void SetSSAO(bool);

private:
    Array1<KiriPointLightPtr> pointLights;
    void Setup() override;

    const float constant = 1.0f;
    const float linear = 0.7f;
    const float quadratic = 1.8f;

    bool b_ssao;
};
typedef SharedPtr<KiriMaterialBlinnDefer> KiriMaterialBlinnDeferPtr;
#endif