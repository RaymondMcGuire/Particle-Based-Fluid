/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:15:55
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_cube_skbr.h
 */

#ifndef _KIRI_MATERIAL_CUBE_SKBR_H_
#define _KIRI_MATERIAL_CUBE_SKBR_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialCubeSKBR : public KiriMaterial
{
public:
    KiriMaterialCubeSKBR();
    KiriMaterialCubeSKBR(UInt, bool = true);

    void Setup() override;
    void Update() override;

    void SetReflection(bool _r)
    {
        reflection = _r;
    }

private:
    UInt cubeSkyboxTexture;
    bool reflection;
};
typedef SharedPtr<KiriMaterialCubeSKBR> KiriMaterialCubeSKBRPtr;
#endif