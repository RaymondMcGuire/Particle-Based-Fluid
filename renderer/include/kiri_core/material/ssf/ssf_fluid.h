/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-24 20:04:09
 * @LastEditTime: 2021-02-20 01:00:08
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\material\ssf\ssf_fluid.h
 */

#ifndef _KIRI_SSF_FLUID_H_
#define _KIRI_SSF_FLUID_H_

#pragma once

#include <kiri_core/material/material.h>

class KiriMaterialSSFFluid : public KiriMaterial
{
public:
    KiriMaterialSSFFluid();

    void Update() override;

    void SetParticleView(bool);

    void SetDepthTex(Int);
    void SetNormalTex(Int);
    void SetThickTex(Int);
    void SetMultiColorTex(Int);
    void SetSkyBoxTex(Int);
    void SetBGTex(Int);
    void SetBgDepthTex(Int);

    void SetMultiColor(bool);
    void SetRenderOpt(Int renderOpt);

    void SetCameraParams(float aspect, float zFar, float zNear, float zFov, Matrix4x4F invMat);

private:
    void Setup() override;

    float mR0, mAspect, mFar, mNear, mFov;
    Matrix4x4F mInvMat;
    bool bMultiColor;
    Int mRenderOpt;
};
typedef SharedPtr<KiriMaterialSSFFluid> KiriMaterialSSFFluidPtr;
#endif