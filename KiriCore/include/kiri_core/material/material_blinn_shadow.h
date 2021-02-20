/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:16:30
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_blinn_shadow.h
 */

#ifndef _KIRI_MATERIAL_BLINN_SHADOW_H_
#define _KIRI_MATERIAL_BLINN_SHADOW_H_
#pragma once

#include <kiri_core/material/material.h>
#include <kiri_core/shadow/shadow_mapping.h>

class KiriMaterialBlinnShadow : public KiriMaterial
{
public:
    KiriMaterialBlinnShadow(KiriShadowMapping *);

    void Setup() override;
    void Update() override;

private:
    UInt texture;
    KiriShadowMapping *shadow;
};
typedef SharedPtr<KiriMaterialBlinnShadow> KiriMaterialBlinnShadowPtr;
#endif