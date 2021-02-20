/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 13:39:35
 * @LastEditTime: 2021-02-20 00:06:28
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_application.h
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

        void PushLayer(KiriLayer *Layer);

        void PopCurrentLayer();
        void PopLayer(KiriLayer *Layer);

        inline static KiriApplication &Get() { return *sInstance; };
        inline KiriWindow &GetWindow() { return *mWindow; };

        inline UnSortedMap<String, KiriLayer *> &ExamplesList() { return mExamplesList; }
        inline String CurrentExampleName() const { return mCurrentExampleName; }

        void SetCurrentExampleName(const String &exampleName) { mCurrentExampleName = exampleName; }
        void AddExample(const String &exampleName, KiriLayer *example) { mExamplesList[exampleName] = example; }

    private:
        bool OnWindowCloseEvent(KiriWindowCloseEvent &e);
        bool OnWindowResizeEvent(KiriWindowResizeEvent &e);

        UniquePtr<KiriWindow> mWindow;

        bool mMinimized = false;
        bool mRunning = true;
        bool mCaptureScreen = false;

        KiriLayerImGui *mLayerImGui;
        KiriLayerStack mLayerStack;
        KiriTimer mTimer;

        String mCurrentExampleName = "";
        UnSortedMap<String, KiriLayer *> mExamplesList;

        static KiriApplication *sInstance;
    };

    typedef SharedPtr<KiriApplication> KiriApplicationPtr;
    KiriApplicationPtr CreateApplication();
} // namespace KIRI

#endif