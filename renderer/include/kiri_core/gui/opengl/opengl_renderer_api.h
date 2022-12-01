/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 12:53:09
 * @LastEditTime: 2020-10-26 02:43:09
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\gui\opengl\opengl_renderer_api.h
 */

#ifndef _KIRI_OPENGL_RENDERER_API_H_
#define _KIRI_OPENGL_RENDERER_API_H_
#pragma once
#include <kiri_core/renderer/renderer_api.h>

namespace KIRI
{
    class KiriOpenGLRendererAPI : public KiriRendererAPI
    {
    public:
        virtual void Init() override;

        virtual void Clear(bool depth) override;
        virtual void SetClearColor(const Vector4F &color) override;
        virtual void SetViewport(const Vector4F &rect) override;

        virtual void GlobalUboGenerate() override;
        virtual void GlobalUboBind(const KiriCameraPtr &camera) override;

    private:
        UInt mUboMatrices;
    };
} // namespace KIRI

#endif