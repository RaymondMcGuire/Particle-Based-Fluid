/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-19 22:25:28
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\material\ssf\ssf_normal.h
 */

#ifndef _KIRI_SSF_NORMAL_H_
#define _KIRI_SSF_NORMAL_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialSSFNormal : public KiriMaterial
{
public:
    KiriMaterialSSFNormal();
    void Update() override;

    void SetDepthTex(Int);
    void SetParticleScale(float);

private:
    float mParticleScale = 1.f;
    void Setup() override;
};
typedef SharedPtr<KiriMaterialSSFNormal> KiriMaterialSSFNormalPtr;
#endif