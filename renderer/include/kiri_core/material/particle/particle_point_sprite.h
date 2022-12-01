/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:18:15
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\particle\particle_point_sprite.h
 */

#ifndef _KIRI_MATERIAL_PARTICLE_POINT_SPRITE_H_
#define _KIRI_MATERIAL_PARTICLE_POINT_SPRITE_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialParticlePointSprite : public KiriMaterial
{
public:
    KiriMaterialParticlePointSprite();

    void Update() override;

    void SetParticleScale(float);
    void SetParticleRadius(float);

private:
    float mParticleScale = 1.f;
    float mParticleRadius = 1.f;
    Vector3F mBaseColor = Vector3F(0.1f, 0.1f, 0.8f);

    void Setup() override;
};

typedef SharedPtr<KiriMaterialParticlePointSprite> KiriMaterialParticlePointSpritePtr;
#endif