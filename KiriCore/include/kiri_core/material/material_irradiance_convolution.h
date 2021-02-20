/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:14:27
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_irradiance_convolution.h
 */

#ifndef _KIRI_MATERIAL_IRRADIANCE_CONVOLUTION_H_
#define _KIRI_MATERIAL_IRRADIANCE_CONVOLUTION_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialIrradianceConvolution : public KiriMaterial
{
public:
    KiriMaterialIrradianceConvolution(Matrix4x4F);

    void Setup() override;
    void Update() override;

private:
    Matrix4x4F mCaptureProjection;
};
typedef SharedPtr<KiriMaterialIrradianceConvolution> KiriMaterialIrradianceConvolutionPtr;
#endif