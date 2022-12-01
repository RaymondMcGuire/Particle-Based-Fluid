/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 12:25:41
 * @LastEditTime: 2021-02-20 19:39:27
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\renderer\renderer_api.h
 */

#ifndef _KIRI_RENDERER_API_H_
#define _KIRI_RENDERER_API_H_
#pragma once
#include <kiri_pch.h>
#include <kiri_core/camera/camera.h>

namespace KIRI
{
    class KiriRendererAPI
    {
    public:
        enum class RenderPlatform
        {
            None = 0,
            OpenGL = 1
        };

    public:
        virtual void Init() = 0;

        virtual void Clear(bool depth) = 0;
        virtual void SetClearColor(const Vector4F &color) = 0;
        virtual void SetViewport(const Vector4F &rect) = 0;

        // Uniform Buffer Objects
        virtual void GlobalUboGenerate() = 0;
        virtual void GlobalUboBind(const KiriCameraPtr &camera) = 0;

        inline static RenderPlatform GetRenderPlatform() { return sRenderPlatform; }

    private:
        static RenderPlatform sRenderPlatform;
    };
} // namespace KIRI

#endif