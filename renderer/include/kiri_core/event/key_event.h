/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-19 01:56:14
 * @LastEditTime: 2021-02-20 19:45:06
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\event\key_event.h
 */
#ifndef _KIRI_KEY_EVENT_H_
#define _KIRI_KEY_EVENT_H_
#pragma once
#include <kiri_core/event/event.h>

namespace KIRI
{

    class KiriKeyEvent : public KiriEvent
    {
    public:
        inline Int GetKeyCode() { return mKeyCode; };

        EVENT_CLASS_CATEGORY(EventCategoryKeyboard | EventCategoryInput)
    protected:
        KiriKeyEvent(Int keycode)
            : mKeyCode(keycode) {}

        Int mKeyCode;
    };

    class KiriKeyPressedEvent : public KiriKeyEvent
    {
    public:
        KiriKeyPressedEvent(Int keyCode, Int repeatCount)
            : KiriKeyEvent(keyCode), mRepeatCount(repeatCount) {}

        inline Int GetRepeatCount() { return mRepeatCount; };

        String ToString() const override
        {
            std::stringstream ss;
            ss << "KeyPressedEvent: " << mKeyCode << "(" << mRepeatCount << " Repeats)";
            return ss.str();
        }

        EVENT_CLASS_TYPE(KeyPressed)
    private:
        Int mRepeatCount;
    };

    class KiriKeyReleasedEvent : public KiriKeyEvent
    {
    public:
        KiriKeyReleasedEvent(Int keyCode)
            : KiriKeyEvent(keyCode) {}

        String ToString() const override
        {
            std::stringstream ss;
            ss << "KeyReleasedEvent: " << mKeyCode;
            return ss.str();
        }

        EVENT_CLASS_TYPE(KeyReleased)
    };

    class KiriKeyTypeEvent : public KiriKeyEvent
    {
    public:
        KiriKeyTypeEvent(Int keyCode)
            : KiriKeyEvent(keyCode) {}

        String ToString() const override
        {
            std::stringstream ss;
            ss << "KeyTypeEvent: " << mKeyCode;
            return ss.str();
        }

        EVENT_CLASS_TYPE(KeyType)
    };

} // namespace KIRI

#endif