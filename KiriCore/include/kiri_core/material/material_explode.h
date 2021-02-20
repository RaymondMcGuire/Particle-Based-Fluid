/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:15:24
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_explode.h
 */

#ifndef _KIRI_MATERIAL_EXPLODE_H_
#define _KIRI_MATERIAL_EXPLODE_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialExplode : public KiriMaterial
{
public:
    KiriMaterialExplode();

    void Setup() override;
    void Update() override;
};
typedef SharedPtr<KiriMaterialExplode> KiriMaterialExplodePtr;
#endif