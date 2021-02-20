/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:16:55
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_blinn_point_shadow.h
 */

#ifndef _KIRI_MATERIAL_BLINN_POINT_SHADOW_H_
#define _KIRI_MATERIAL_BLINN_POINT_SHADOW_H_
#pragma once
#include <kiri_core/texture/texture.h>
#include <kiri_core/material/material.h>
#include <kiri_core/shadow/point_shadow.h>

class KiriMaterialBlinnPointShadow : public KiriMaterial
{
public:
    KiriMaterialBlinnPointShadow(bool, KiriPointShadow *, KiriTexture);

    void Setup() override;
    void Update() override;

private:
    UInt texture;
    bool outside;
    KiriPointShadow *shadow;
};
typedef SharedPtr<KiriMaterialBlinnPointShadow> KiriMaterialBlinnPointShadowPtr;
#endif