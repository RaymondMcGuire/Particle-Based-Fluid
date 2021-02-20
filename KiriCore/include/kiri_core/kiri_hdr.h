/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:40:54
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\kiri_hdr.h
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

    void enable();
    void bindHDR();

    void renderBloom();
    void SetBloom(bool);
    void SetExposure(float);
    void SetHDR(bool);

    void release();
    void renderToScreen();

    UInt getSceneBuffer() { return colorBuffers[0]; }

private:
    UInt WINDOW_WIDTH, WINDOW_HEIGHT;

    UInt hdrFBO;
    UInt colorBuffers[2];
    UInt attachments[2];
    UInt rboDepth;

    bool bloom;
    UInt blurFBO[2];
    UInt blurColorbuffers[2];

    KiriMaterialHDRPtr hdrMaterial;
    KiriMaterialGaussianBlurPtr gaussBlurMaterial;
    KiriQuadPtr quad;
};
#endif