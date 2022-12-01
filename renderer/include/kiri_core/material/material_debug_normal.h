/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:15:42
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_debug_normal.h
 */

#ifndef _KIRI_MATERIAL_DEBUG_NORMAL_H_
#define _KIRI_MATERIAL_DEBUG_NORMAL_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialDebugNomral : public KiriMaterial
{
public:
    KiriMaterialDebugNomral();

    void Setup() override;
    void Update() override;
};
typedef SharedPtr<KiriMaterialDebugNomral> KiriMaterialDebugNomralPtr;
#endif