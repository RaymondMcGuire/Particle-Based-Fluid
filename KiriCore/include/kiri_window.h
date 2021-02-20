/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 12:08:43
 * @LastEditTime: 2021-02-19 22:37:24
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_window.h
 */
#ifndef _KIRI_WINDOW_H_
#define _KIRI_WINDOW_H_

#pragma once

#include <kiri_core/event/event.h>

namespace KIRI
{
    struct WindowProperty
    {
        String Title;
        UInt Width;
        UInt Height;
        bool LockFPS;

        // Constructor
        WindowProperty(String title = "KIRI", UInt width = 1920, UInt height = 1080, bool lockFPS = true)
            : Title(title), Width(width), Height(height), LockFPS(lockFPS) {}
    };

    class KiriWindow
    {
    public:
        using EventCallbackFunc = std::function<void(KiriEvent &)>;

        virtual ~KiriWindow(){};

        virtual void OnUpdate() = 0;

        virtual UInt GetWindowWidth() const = 0;
        virtual UInt GetWindowHeight() const = 0;
        virtual bool IsFullscreen() const = 0;

        virtual void SetEventCallback(const EventCallbackFunc &callback) = 0;
        virtual void LockFPS(bool enabled) = 0;
        virtual bool IsLocked() const = 0;

        virtual void Init(const WindowProperty &windowProperty) = 0;
        virtual void ShutDown() = 0;

        virtual void *GetNativeWindow() const = 0;

        static KiriWindow *CreateKIRIWindow(const WindowProperty &property = WindowProperty());
    };
} // namespace KIRI

#endif