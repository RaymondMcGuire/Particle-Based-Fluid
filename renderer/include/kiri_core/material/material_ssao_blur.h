/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:03:47
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_ssao_blur.h
 */

#ifndef _KIRI_MATERIAL_SSAO_BLUR_H_
#define _KIRI_MATERIAL_SSAO_BLUR_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialSSAOBlur : public KiriMaterial
{
public:
    KiriMaterialSSAOBlur();
    void Update() override;

private:
    void Setup() override;
};
typedef SharedPtr<KiriMaterialSSAOBlur> KiriMaterialSSAOBlurPtr;
#endif