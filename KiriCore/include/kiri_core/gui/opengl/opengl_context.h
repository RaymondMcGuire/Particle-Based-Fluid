/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 12:53:09
 * @LastEditTime: 2020-10-25 19:09:18
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\gui\opengl\opengl_context.h
 */

#ifndef _KIRI_OPENGL_CONTEXT_H_
#define _KIRI_OPENGL_CONTEXT_H_
#pragma once
#include <kiri_core/renderer/renderer_context.h>

struct GLFWwindow;

namespace KIRI
{
    class KiriOpenGLContext : public KiriRendererContext
    {
    public:
        KiriOpenGLContext(GLFWwindow *windowHandle);

        virtual void Init() override;
        virtual void SwapBuffers() override;

    private:
        GLFWwindow *mWindowHandle;
    };
} // namespace KIRI

#endif