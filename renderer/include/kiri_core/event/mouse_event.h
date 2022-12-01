/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-19 01:56:14
 * @LastEditTime: 2020-10-25 03:18:21
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\event\mouse_event.h
 */
#ifndef _KIRI_MOUSE_EVENT_H_
#define _KIRI_MOUSE_EVENT_H_
#pragma once
#include <kiri_core/event/event.h>

namespace KIRI
{

    class KiriMouseMoveEvent : public KiriEvent
    {
    public:
        KiriMouseMoveEvent(float mousePosX, float mousePosY)
            : mMouseX(mousePosX), mMouseY(mousePosY) {}

        inline float GetX() { return mMouseX; };
        inline float GetY() { return mMouseY; };

        String ToString() const override
        {
            std::stringstream ss;
            ss << "MouseMovedEvent: (" << mMouseX << "," << mMouseY << ")";
            return ss.str();
        }

        EVENT_CLASS_CATEGORY(EventCategoryInput | EventCategoryMouse)
        EVENT_CLASS_TYPE(MouseMoved)
    private:
        float mMouseX, mMouseY;
    };

    class KiriMouseScrollEvent : public KiriEvent
    {
    public:
        KiriMouseScrollEvent(float mouseXOffset, float mouseYOffset)
            : mMouseXOffset(mouseXOffset), mMouseYOffset(mouseYOffset) {}

        inline float GetMouseXOffset() { return mMouseXOffset; };
        inline float GetMouseYOffset() { return mMouseYOffset; };

        String ToString() const override
        {
            std::stringstream ss;
            ss << "MouseScrollEvent Offset: (" << mMouseXOffset << "," << mMouseYOffset << ")";
            return ss.str();
        }

        EVENT_CLASS_CATEGORY(EventCategoryInput | EventCategoryMouse)
        EVENT_CLASS_TYPE(MouseScrolled)
    private:
        float mMouseXOffset, mMouseYOffset;
    };

    class KiriMouseButtonEvent : public KiriEvent
    {
    public:
        inline Int GetMouseButton() const { return mMouseButton; };

        EVENT_CLASS_CATEGORY(EventCategoryInput | EventCategoryMouse)

    protected:
        KiriMouseButtonEvent(Int mouseButton)
            : mMouseButton(mouseButton) {}

        Int mMouseButton;
    };

    class KiriMouseButtonPressedEvent : public KiriMouseButtonEvent
    {
    public:
        KiriMouseButtonPressedEvent(Int button)
            : KiriMouseButtonEvent(button) {}

        String ToString() const override
        {
            std::stringstream ss;
            ss << "MouseButtonPressedEvent: " << mMouseButton;
            return ss.str();
        }

        EVENT_CLASS_TYPE(MouseButtonPressed)
    };

    class KiriMouseButtonReleasedEvent : public KiriMouseButtonEvent
    {
    public:
        KiriMouseButtonReleasedEvent(Int button)
            : KiriMouseButtonEvent(button) {}

        String ToString() const override
        {
            std::stringstream ss;
            ss << "MouseButtonReleasedEvent: " << mMouseButton;
            return ss.str();
        }

        EVENT_CLASS_TYPE(MouseButtonReleased)
    };

} // namespace KIRI

#endif