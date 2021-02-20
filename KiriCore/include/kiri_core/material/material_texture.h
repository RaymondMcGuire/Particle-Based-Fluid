/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:01:50
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_texture.h
 */

#ifndef _KIRI_MATERIAL_TEXTURE_H_
#define _KIRI_MATERIAL_TEXTURE_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialTexture : public KiriMaterial
{
public:
    KiriMaterialTexture();

    void Setup() override;
    void Update() override;
};

typedef SharedPtr<KiriMaterialTexture> KiriMaterialTexturePtr;
#endif