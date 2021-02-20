/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:17:57
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\particle\particle_instance.h
 */

#ifndef _KIRI_MATERIAL_PARTICLE_INSTANCE_H_
#define _KIRI_MATERIAL_PARTICLE_INSTANCE_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialParticleInstance : public KiriMaterial
{
public:
    KiriMaterialParticleInstance();
    KiriMaterialParticleInstance(bool singleColor, Array1Vec4F colorArray);

    void SetDirLightPos(Vector3F);
    void SetParticleColor(Vector3F);
    void SetLightColor(Vector3F);

    void updateColorArray(Array1Vec4F colorArray);

    void Setup() override;
    void Update() override;

private:
    Vector3F _particleColor = Vector3F(35.0f / 255.0f, 137.0f / 255.0f, 218.0f / 255.0f);
    Vector3F _dirLightPos = Vector3F(1.2f, 1.0f, 2.0f);
    Vector3F _lightColor = Vector3F(1.0f, 1.0f, 1.0f);

    bool _singleColor;
    Array1Vec4F _colorArray;
    UInt color_tbo, color_buffer;
};
typedef SharedPtr<KiriMaterialParticleInstance> KiriMaterialParticleInstancePtr;
#endif