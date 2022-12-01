/***
 * @Author: Jayden Zhang
 * @Date: 2020-10-19 01:56:14
 * @LastEditTime: 2020-10-25 03:18:39
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriCore\include\kiri_core\event\application_event.h
 */
#ifndef _KIRI_APPLICATION_EVENT_H_
#define _KIRI_APPLICATION_EVENT_H_
#pragma once
#include <kiri_core/event/event.h>

namespace KIRI
{

  class KiriWindowResizeEvent : public KiriEvent
  {
  public:
    KiriWindowResizeEvent(Int mWidth, Int height)
        : mWidth(mWidth), mHeight(height) {}

    inline Int GetWindowWidth() const { return mWidth; };
    inline Int GetWindowHeight() const { return mHeight; };

    String ToString() const override
    {
      std::stringstream ss;
      ss << "WindowResizeEvent: New Window size: (" << mWidth << "," << mHeight
         << ")";
      return ss.str();
    }

    EVENT_CLASS_TYPE(WindowResize)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
  private:
    UInt mWidth, mHeight;
  };

  class KiriWindowCloseEvent : public KiriEvent
  {
  public:
    KiriWindowCloseEvent() {}

    EVENT_CLASS_TYPE(WindowClose)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
  };

  class KiriAppTickEvent : public KiriEvent
  {
  public:
    KiriAppTickEvent() {}

    EVENT_CLASS_TYPE(AppTick)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
  };

  class KiriAppUpdateEvent : public KiriEvent
  {
  public:
    KiriAppUpdateEvent() {}

    EVENT_CLASS_TYPE(AppUpdate)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
  };

  class KiriAppRenderEvent : public KiriEvent
  {
  public:
    KiriAppRenderEvent() {}

    EVENT_CLASS_TYPE(AppRender)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
  };

} // namespace KIRI

#endif