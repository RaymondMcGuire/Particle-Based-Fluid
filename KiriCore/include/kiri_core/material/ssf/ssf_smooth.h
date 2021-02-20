/*** 
 * @Author: Xu.WANG
 * @Date: 2020-06-16 01:32:27
 * @LastEditTime: 2021-02-20 18:59:16
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\ssf\ssf_smooth.h
 */

#ifndef _KIRI_SSF_SMOOTH_H_
#define _KIRI_SSF_SMOOTH_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialSSFSmooth : public KiriMaterial
{
public:
    KiriMaterialSSFSmooth();
    void Update() override;

    void SetSmoothTex(Int id);
    void SetRealDepthTex(Int id);
    void SetMultiColorTex(Int id);
    void SetParticleView(bool enable_particle_view);
    void SetEnableSSF(bool enable_ssf);

private:
    void Setup() override;

    Int mkernelR;
    float mSigmaR;
    float mSigmaZ;
};
typedef SharedPtr<KiriMaterialSSFSmooth> KiriMaterialSSFSmoothPtr;
#endif