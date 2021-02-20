/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:41:13
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\kiri_ssao.h
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
    void enable();
    void render(UInt, UInt);

    UInt getSSAOColorBuffer()
    {
        return ssaoColorBufferBlur;
    }

private:
    UInt WINDOW_WIDTH, WINDOW_HEIGHT;

    UInt ssaoFBO, ssaoBlurFBO;
    UInt ssaoColorBuffer, ssaoColorBufferBlur;

    Array1Vec3F ssaoKernel;
    Array1Vec3F ssaoNoise;
    UInt noiseTexture;
    void sampleKernel();
    void generateNoiseTexure();

    KiriQuadPtr quad;
    KiriMaterialSSAOPtr mSSAO;
    KiriMaterialSSAOBlurPtr mSSAOBlur;

    void ssao(UInt, UInt);
    void blur();
};

#endif