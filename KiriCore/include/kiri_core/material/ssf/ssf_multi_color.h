/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 18:57:46
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\ssf\ssf_multi_color.h
 */

#ifndef _KIRI_SSF_MULTI_COLOR_H_
#define _KIRI_SSF_MULTI_COLOR_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialSSFMultiColor : public KiriMaterial
{
public:
    KiriMaterialSSFMultiColor();
    void Update() override;

    void SetParticleScale(float);
    void SetParticleRadius(float);
    void SetTransparentMode(bool);

private:
    float mParticleScale = 1.f;
    float mParticleRadius = 1.f;
    bool bFluidTransparent = false;

    void Setup() override;
};
typedef SharedPtr<KiriMaterialSSFMultiColor> KiriMaterialSSFMultiColorPtr;
#endif