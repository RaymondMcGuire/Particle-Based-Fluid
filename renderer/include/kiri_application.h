/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2021-02-26 11:37:06
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-06-24 15:49:28
 * @FilePath: \Kiri\KiriCore\include\kiri_application.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _KIRI_APPLICATION_H_
#define _KIRI_APPLICATION_H_
#pragma once
#include <kiri_core/event/application_event.h>
#include <kiri_core/gui/layer_imgui.h>
#include <kiri_core/gui/layer_stack.h>
#include <kiri_window.h>

namespace KIRI
{
  class KiriApplication
  {
  public:
    KiriApplication();
    virtual ~KiriApplication();

    void Run();
    void ShutDown();
    void CaptureScreen(bool Enable);

    void OnEvent(KiriEvent &e);

    void PushLayer(const String &layerName);
    void PopLayer(const String &layerName);
    void PopCurrentLayer();

    inline static KiriApplication &Get() { return *sInstance; };
    inline KiriWindow &GetWindow() { return *mWindow; };

    inline const UnSortedMap<String, KiriLayerPtr> &ExamplesList()
    {
      return mExamplesList;
    }
    inline const String &CurrentExampleName() const
    {
      return mCurrentExampleName;
    }

    void SetCurrentExampleName(const String &exampleName)
    {
      mCurrentExampleName = exampleName;
    }
    void AddExample(const String &exampleName, KiriLayerPtr example)
    {
      mExamplesList[exampleName] = example;
    }
    void RefreshApp() { mLayerImGui->SwitchApp(mCurrentExampleName); }

  private:
    bool OnWindowCloseEvent(KiriWindowCloseEvent &e);
    bool OnWindowResizeEvent(KiriWindowResizeEvent &e);

    UniquePtr<KiriWindow> mWindow;

    bool bMinimized = false;
    bool bRunning = true;
    bool bCaptureScreen = false;

    KiriLayerImGuiPtr mLayerImGui;
    KiriLayerStack mLayerStack;
    KiriTimer mTimer;

    String mCurrentExampleName = "";
    UnSortedMap<String, KiriLayerPtr> mExamplesList;

    static KiriApplication *sInstance;
  };

  typedef SharedPtr<KiriApplication> KiriApplicationPtr;
  KiriApplicationPtr CreateApplication();
} // namespace KIRI

#endif