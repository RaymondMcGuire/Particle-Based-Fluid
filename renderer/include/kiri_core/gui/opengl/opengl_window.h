/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 12:22:17
 * @LastEditTime: 2021-02-20 18:55:13
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\gui\opengl\opengl_window.h
 */

#ifndef _KIRI_OPENGL_WINDOW_H_
#define _KIRI_OPENGL_WINDOW_H_

#pragma once

#include <kiri_window.h>
#include <kiri_core/renderer/renderer_context.h>

#include<glad/glad.h>
#include <GLFW/glfw3.h>


namespace KIRI
{
    class KiriOpenGLWindow : public KiriWindow
    {
    public:
        KiriOpenGLWindow(const WindowProperty &windowProperty);
        virtual ~KiriOpenGLWindow();

        virtual void OnUpdate() override;

        inline virtual UInt GetWindowWidth() const override { return mWindowData.Width; };
        inline virtual UInt GetWindowHeight() const override { return mWindowData.Height; };
        virtual bool IsFullscreen() const override;

        inline virtual void SetEventCallback(const EventCallbackFunc &callback) override { mWindowData.EventCallback = callback; };
        virtual void LockFPS(bool enabled) override;
        virtual bool IsLocked() const override;

        inline virtual void *GetNativeWindow() const override { return mWindow; };

    private:
        virtual void Init(const WindowProperty &windowProperty) override;

        virtual void ShutDown() override;

    private:
        GLFWwindow *mWindow;
        KiriRendererContext *mRenderContext;

        struct WindowData
        {
            String Title;
            UInt Width, Height;
            bool LockFPS;
            EventCallbackFunc EventCallback;
        };

        WindowData mWindowData;
    };
} // namespace KIRI

#endif