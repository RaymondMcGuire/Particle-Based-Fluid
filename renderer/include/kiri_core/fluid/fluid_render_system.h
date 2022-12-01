/***
 * @Author: Xu.WANG
 * @Date: 2020-10-24 20:04:09
 * @LastEditTime: 2021-02-20 19:00:11
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \KiriCore\include\kiri_core\fluid\fluid_render_system.h
 */

#ifndef _KIRI_FLUID_RENDER_SYSTEM_H_
#define _KIRI_FLUID_RENDER_SYSTEM_H_

#pragma once

#include <kiri_core/material/ssf/ssf_depth.h>
#include <kiri_core/material/ssf/ssf_thick.h>
#include <kiri_core/material/ssf/ssf_normal.h>
#include <kiri_core/material/ssf/ssf_multi_color.h>
#include <kiri_core/material/ssf/ssf_smooth.h>
#include <kiri_core/material/ssf/ssf_fluid.h>

#include <kiri_core/kiri_framebuffer.h>

#include <kiri_core/camera/camera.h>

namespace KIRI
{
    class KiriFluidRenderSystem
    {
    public:
        KiriFluidRenderSystem() {}
        KiriFluidRenderSystem(UInt mWidth, UInt height, KiriCameraPtr camera);

        size_t NumOfParticles() { return mNumOfParticles; }

        void SetSkyBoxTex(UInt);
        void EnableFluidTransparentMode(bool);
        void EnableSoildSsfMode(bool);

        void SetParticles(Array1Vec3F, float);
        void SetParticlesWithRadius(Array1Vec4F pos, Array1Vec4F col, UInt num);
        void SetParticlesVBO(UInt vbo, UInt num, float radius);
        void SetParticlesVBOWithRadius(UInt pvbo, UInt cvbo, UInt num);
        void SetParticlesVBO(UInt pvbo, UInt cvbo, UInt num, float radius);
        void renderFluid(UInt, UInt);

    private:
        UInt SCREEN_WIDTH, SCREEN_HEIGHT;
        UInt mSkyBoxTex;

        KiriCameraPtr mCamera;

        // particles param
        UInt mParticlesVAO;
        UInt mParticlesVBO;
        UInt mParticlesColorVBO;

        size_t mNumOfParticles;
        float mParticleRadius;
        Int mSmoothIter;

        // remder params
        bool bMultiColor;

        // fluid FBO
        UInt fluidFBO;

        UInt depthTex;
        // GL_RED32F
        UInt depthATex, depthBTex;
        UInt thickTex;

        // GL_RGBA32F
        UInt normTex;
        UInt multiColorTex;

        // material
        KiriMaterialSSFDepthPtr mDepthShader;
        KiriMaterialSSFThickPtr mThickShader;
        KiriMaterialSSFNormalPtr mNormalShader;
        KiriMaterialSSFMultiColorPtr mMultiColorShader;
        KiriMaterialSSFFluidPtr mFluidShader;
        KiriMaterialSSFSmoothPtr mSmoothShader;

        // init
        void InitBuffer();

        // Render tex method
        void RenderDepthTex();
        void RenderThickTex();
        void RenderNormalTex();
        void RenderMultiColorTex();
        void RenderFluidTex(UInt, UInt);
        void RenderSmoothTex(bool);

        // helper method
        float CalcParticleScale();

        // mQuad
        UInt mQuadVAO;
        void InitQuadBuff();

        bool bDepthAB;
        bool bFluidTransparent;
        bool bSoildSsf;
        // mSmooth tex
        inline UInt smoothTex() { return bDepthAB ? depthATex : depthBTex; }
        // Render tex
        inline UInt realDepthTex() { return bDepthAB ? depthBTex : depthATex; }
    };

    typedef SharedPtr<KiriFluidRenderSystem> KiriFluidRenderSystemPtr;
}
#endif