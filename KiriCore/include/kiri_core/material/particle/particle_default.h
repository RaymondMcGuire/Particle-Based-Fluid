/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:27:47
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\particle\particle_default.h
 */

#ifndef _KIRI_MATERIAL_PARTICLE_DEFAULT_H_
#define _KIRI_MATERIAL_PARTICLE_DEFAULT_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialParticleDefault : public KiriMaterial
{
public:
    KiriMaterialParticleDefault();

    void SetParticleColor(Vector3F);
    void Update() override;

private:
    void Setup() override;

private:
    Vector3F particle_color = Vector3F(100.0f, 0.0f, 0.0f);
};
typedef SharedPtr<KiriMaterialParticleDefault> KiriMaterialParticleDefaultPtr;
#endif