/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 14:11:09
 * @LastEditTime: 2021-02-20 19:45:37
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\gui\opengl\opengl_input.h
 */
#ifndef _KIRI_OPENGL_INPUT_H_
#define _KIRI_OPENGL_INPUT_H_
#pragma once
#include <kiri_input.h>

namespace KIRI
{
    class KiriOpenGLInput : public KiriInput
    {
    protected:
        virtual bool IsKeyDownImpl(int keycode) override;
        virtual bool IsMouseButtonDownImpl(int mouseButton) override;
        virtual Vector2F GetMousePosImpl() override;
        virtual float GetMouseXImpl() override;
        virtual float GetMouseYImpl() override;
    };

} // namespace KIRI

#endif