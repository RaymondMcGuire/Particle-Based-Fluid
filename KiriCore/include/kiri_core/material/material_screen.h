/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:01:09
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_screen.h
 */

#ifndef _KIRI_MATERIAL_SCREEN_H_
#define _KIRI_MATERIAL_SCREEN_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialScreen : public KiriMaterial
{
public:
    KiriMaterialScreen();

    void Setup() override;
    void Update() override;

    void SetPostProcessingType(Int type);

private:
    Int mPostProcessingType;
};
typedef SharedPtr<KiriMaterialScreen> KiriMaterialScreenPtr;
#endif