/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 13:56:06
 * @FilePath: \core\include\kiri_core\kiri_hdr.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _KIRI_HDR_H_
#define _KIRI_HDR_H_
#pragma once
#include <kiri_pch.h>
#include <kiri_core/model/model_quad.h>
#include <kiri_core/material/material_hdr.h>
#include <kiri_core/material/material_gaussian_blur.h>
class KiriHDR
{
public:
    KiriHDR(UInt, UInt, bool);

    void Enable();
    void BindHDR();

    void RenderBloom();
    void SetBloom(bool);
    void SetExposure(float);
    void SetHDR(bool);

    void Release();
    void RenderToScreen();

    UInt GetSceneBuffer() { return mColorBuffers[0]; }

private:
    UInt WINDOW_WIDTH, WINDOW_HEIGHT;

    UInt mHDRFBO;
    UInt mColorBuffers[2];
    UInt mAttachments[2];
    UInt mRBODepth;

    bool mEnableBloom;
    UInt mBlurFBO[2];
    UInt mBlurColorBuffers[2];

    KiriMaterialHDRPtr mHDRMaterial;
    KiriMaterialGaussianBlurPtr mGaussBlurMaterial;
    KiriQuadPtr mQuad;
};
#endif