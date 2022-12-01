/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:15:17
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_g_buffer.h
 */

#ifndef _KIRI_MATERIAL_G_BUFFER_H_
#define _KIRI_MATERIAL_G_BUFFER_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialGBuffer : public KiriMaterial
{
public:
    KiriMaterialGBuffer();
    void Update() override;

    void SetHaveNormalMap(bool);
    void SetUseNormalMap(bool);
    void SetOutside(bool);

private:
    void Setup() override;
    bool use_normal;
    bool have_normal;
    bool outside;
};
typedef SharedPtr<KiriMaterialGBuffer> KiriMaterialGBufferPtr;
#endif