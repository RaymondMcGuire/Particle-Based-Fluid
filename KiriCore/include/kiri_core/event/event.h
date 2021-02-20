/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-09-27 02:54:00
 * @LastEditTime: 2020-10-25 12:48:59
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\event\event.h
 */

#ifndef _KIRI_EVENT_H_
#define _KIRI_EVENT_H_
#pragma once
#include <kiri_pch.h>

namespace KIRI
{
    enum class EventType
    {
        None = 0,
        WindowClose,
        WindowResize,
        WindowFocus,
        WindowLostFocus,
        WindowMoved,
        AppTick,
        AppUpdate,
        AppRender,
        KeyPressed,
        KeyReleased,
        KeyType,
        MouseButtonPressed,
        MouseButtonReleased,
        MouseMoved,
        MouseScrolled
    };

    enum EventCategory
    {
        None = 0,
        EventCategoryApplication,
        EventCategoryInput,
        EventCategoryKeyboard,
        EventCategoryMouse,
        EventCategoryMouseButton
    };

#define EVENT_CLASS_TYPE(type)                                                  \
    static EventType GetStaticType() { return EventType::##type; }              \
    virtual EventType GetEventType() const override { return GetStaticType(); } \
    virtual const char *GetName() const override { return #type; }
#define EVENT_CLASS_CATEGORY(category) \
    virtual Int GetCategoryFlags() const override { return category; }

#define EVENT_BIND_FUNCTION(function) [&](auto &e) { return this->function(e); }

    class KiriEvent
    {
    public:
        virtual EventType GetEventType() const = 0;
        virtual const char *GetName() const = 0;
        virtual Int GetCategoryFlags() const = 0;
        virtual String ToString() const { return GetName(); };

        inline virtual bool IsInCategory(EventCategory category)
        {
            return GetCategoryFlags() & category;
        }

    public:
        bool mHandled = false;
    };

    class KiriEventDispatcher
    {
    public:
        /*** 
         * @description: Bind the event to this Dispatcher,
         */
        KiriEventDispatcher(KiriEvent &event)
            : mEvent(event) {}

        /*** 
         * @description: Dispatch the event
         * @param {F:Function} 
         * @return {bool} 
         */
        template <typename T, typename F>
        bool DispatchEvent(F &&Func)
        {
            // Check if the Event Type of the Event bond to this dispatcher
            // matches with the function we are using.
            // Because if they are not the same type, then we cannot guarantee
            // that the function is actually valid.
            if (mEvent.GetEventType() == T::GetStaticType())
            {
                // Whatever it is, this certain function of this Event will be called,
                // And the function will decide whether to consume the event or not.
                mEvent.mHandled = Func(*(T *)&mEvent);
                return true;
            }
            return false;
        }

    private:
        KiriEvent &mEvent;
    };

    inline std::ostream &operator<<(std::ostream &os, const KiriEvent &event)
    {
        return os << event.ToString();
    };

} // namespace KIRI

#endif