/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-24 20:04:09
 * @LastEditTime: 2021-02-20 01:09:08
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\particle\particle_render_system.h
 */

#ifndef _KIRI_PARTICLE_RENDER_SYSTEM_H_
#define _KIRI_PARTICLE_RENDER_SYSTEM_H_

#pragma once

#include <kiri_pch.h>
#include <kiri_core/material/particle/particle_point_sprite.h>
#include <kiri_core/camera/camera.h>

namespace KIRI
{
    class KiriParticleRenderSystem
    {
    public:
        KiriParticleRenderSystem() {}
        KiriParticleRenderSystem(KiriCameraPtr camera);

        void SetParticlesVBO(UInt mVBO, UInt Num, float Radius);
        void SetParticles(ArrayAccessor1<Vector3F> Particles, float Radius);
        void SetParticles(ArrayAccessor1<Vector4F> Particles);
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

}
#endif