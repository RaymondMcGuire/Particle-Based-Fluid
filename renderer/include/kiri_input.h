/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-25 14:09:09
 * @LastEditTime: 2021-02-20 19:41:42
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_input.h
 */

#ifndef _KIRI_INPUT_H_
#define _KIRI_INPUT_H_
#pragma once
#include <kiri_pch.h>

namespace KIRI
{
    class KiriInput
    {
    public:
        inline static bool IsKeyDown(int keycode) { return s_Instance->IsKeyDownImpl(keycode); };
        inline static bool IsMouseButtonDown(int mouseButton) { return s_Instance->IsMouseButtonDownImpl(mouseButton); };
        inline static Vector2F GetMousePos() { return s_Instance->GetMousePosImpl(); };
        inline static float GetMouseX() { return s_Instance->GetMouseXImpl(); };
        inline static float GetMouseY() { return s_Instance->GetMouseYImpl(); };

    protected:
        virtual bool IsKeyDownImpl(int keycode) = 0;
        virtual bool IsMouseButtonDownImpl(int mouseButton) = 0;
        virtual Vector2F GetMousePosImpl() = 0;
        virtual float GetMouseXImpl() = 0;
        virtual float GetMouseYImpl() = 0;

    private:
        static KiriInput *s_Instance;
    };
} // namespace KIRI

#endif