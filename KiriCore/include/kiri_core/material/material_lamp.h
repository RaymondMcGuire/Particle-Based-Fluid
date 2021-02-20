/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:14:20
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_lamp.h
 */

#ifndef _KIRI_MATERIAL_LAMP_H_
#define _KIRI_MATERIAL_LAMP_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialLamp : public KiriMaterial
{
public:
    KiriMaterialLamp();
    KiriMaterialLamp(Vector3F);

    void Setup() override;
    void Update() override;

    void SetColor(Vector3F);

private:
    Vector3F lightColor;
};

typedef SharedPtr<KiriMaterialLamp> KiriMaterialLampPtr;
#endif