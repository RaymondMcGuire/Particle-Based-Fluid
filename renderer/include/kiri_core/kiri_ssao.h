/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 13:45:42
 * @FilePath: \core\include\kiri_core\kiri_ssao.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _KIRI_SSAO_H_
#define _KIRI_SSAO_H_
#pragma once
#include <random>

#include <kiri_pch.h>

#include <kiri_core/model/model_quad.h>
#include <kiri_core/material/material_ssao.h>
#include <kiri_core/material/material_ssao_blur.h>

class KiriSSAO
{
public:
    KiriSSAO(UInt, UInt);
    ~KiriSSAO();
    void Enable();
    void Render(UInt, UInt);

    UInt GetSSAOColorBuffer()
    {
        return mSSAOColorBufferBlur;
    }

private:
    UInt WINDOW_WIDTH, WINDOW_HEIGHT;

    UInt mSSAOFBO, mSSAOBlurFBO;
    UInt mSSAOColorBuffer, mSSAOColorBufferBlur;

    Array1Vec3F mSSAOKernel;
    Array1Vec3F mSSAONoise;
    UInt mNoiseTexture;

    KiriQuadPtr mQuad;
    KiriMaterialSSAOPtr mSSAO;
    KiriMaterialSSAOBlurPtr mSSAOBlur;

    void InitSSAO(UInt, UInt);
    void Blur();

    void SampleKernel();
    void GenerateNoiseTexure();
};

#endif