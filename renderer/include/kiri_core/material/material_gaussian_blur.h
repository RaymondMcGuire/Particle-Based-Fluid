/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:15:09
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_gaussian_blur.h
 */

#ifndef _KIRI_MATERIAL_GAUSSIAN_BLUR_H_
#define _KIRI_MATERIAL_GAUSSIAN_BLUR_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialGaussianBlur : public KiriMaterial
{
public:
    KiriMaterialGaussianBlur(UInt);
    void Update() override;

    void SetHorizontal(bool);

private:
    void Setup() override;
    UInt colorBuffer;
};
typedef SharedPtr<KiriMaterialGaussianBlur> KiriMaterialGaussianBlurPtr;
#endif