/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:27:09
 * @FilePath: \core\include\kiri_core\particle\particle_render_system.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _KIRI_PARTICLE_RENDER_SYSTEM_H_
#define _KIRI_PARTICLE_RENDER_SYSTEM_H_

#pragma once

#include <kiri_core/camera/camera.h>
#include <kiri_core/material/particle/particle_point_sprite.h>
#include <kiri_pch.h>

namespace KIRI
{
  class KiriParticleRenderSystem
  {
  public:
    KiriParticleRenderSystem() {}
    KiriParticleRenderSystem(KiriCameraPtr camera);

    void SetParticlesVBO(UInt vbo, UInt num, float radius);
    void SetParticles(Vec_Vec4F particles);
    void SetParticles(Array1<Vector4F> particles);

    void RenderParticles();

    size_t NumOfParticles() { return mNumOfParticles; }

  private:
    float CalcParticleScale();

    size_t mNumOfParticles;
    float mParticleRadius;
    KiriMaterialParticlePointSpritePtr mPointSpriteMaterial;
    KiriCameraPtr mCamera;

    UInt mParticlesVBO;
    UInt mParticlesVAO;
  };

  typedef SharedPtr<KiriParticleRenderSystem> KiriParticleRenderSystemPtr;

} // namespace KIRI
#endif