/*** 
 * @Author: Xu.WANG
 * @Date: 2020-06-16 01:32:27
 * @LastEditTime: 2021-02-20 18:47:04
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\ssf\ssf_depth.h
 */

#ifndef _KIRI_SSF_DEPTH_H_
#define _KIRI_SSF_DEPTH_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialSSFDepth : public KiriMaterial
{
public:
    KiriMaterialSSFDepth();
    void Update() override;

    void SetParticleScale(float);
    void SetParticleRadius(float);

private:
    float mParticleScale = 1.f;
    float mParticleRadius = 1.f;

    void Setup() override;
};
typedef SharedPtr<KiriMaterialSSFDepth> KiriMaterialSSFDepthPtr;
#endif