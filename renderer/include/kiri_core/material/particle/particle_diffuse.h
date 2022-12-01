/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:17:48
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\particle\particle_diffuse.h
 */

#ifndef _KIRI_MATERIAL_PARTICLE_DIFFUSE_H_
#define _KIRI_MATERIAL_PARTICLE_DIFFUSE_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialParticleDiffuse : public KiriMaterial
{
public:
    KiriMaterialParticleDiffuse();

    void SetLightDirection(Vector3F);
    void SetParticleColor(Vector3F);
    void Setup() override;
    void Update() override;

private:
    Vector3F particle_color;
    Vector3F light_direction;
};
typedef SharedPtr<KiriMaterialParticleDiffuse> KiriMaterialParticleDiffusePtr;
#endif