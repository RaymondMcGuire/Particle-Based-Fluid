/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:15:50
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_cube_skybox.h
 */

#ifndef _KIRI_MATERIAL_CUBE_SKYBOX_H_
#define _KIRI_MATERIAL_CUBE_SKYBOX_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialCubeSkybox : public KiriMaterial
{
public:
    KiriMaterialCubeSkybox();

    void Setup() override;
    void Update() override;
};
typedef SharedPtr<KiriMaterialCubeSkybox> KiriMaterialCubeSkyboxPtr;
#endif