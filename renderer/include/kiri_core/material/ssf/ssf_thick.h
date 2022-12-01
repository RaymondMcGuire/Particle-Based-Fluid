/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 18:59:36
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\ssf\ssf_thick.h
 */

#ifndef _KIRI_SSF_THICK_H_
#define _KIRI_SSF_THICK_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialSSFThick : public KiriMaterial
{
public:
    KiriMaterialSSFThick();
    void Update() override;

    void SetParticleScale(float);
    void SetParticleRadius(float);

private:
    float mParticleScale = 1.f;
    float mParticleRadius = 1.f;

    void Setup() override;
};
typedef SharedPtr<KiriMaterialSSFThick> KiriMaterialSSFThickPtr;
#endif