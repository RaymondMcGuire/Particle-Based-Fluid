/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 13:43:57
 * @LastEditTime: 2021-02-18 20:30:50
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_application.cpp
 */

#include <kiri_application.h>
#include <kiri_core/renderer/renderer.h>
#include <kiri_utils.h>

namespace KIRI
{
    KiriApplication *KiriApplication::sInstance = nullptr;

    KiriApplication::KiriApplication()
    {
        KIRI_ASSERT(!sInstance);
        sInstance = this;

        // Create default KIRI Window
        mWindow = UniquePtr<KiriWindow>(KiriWindow::CreateKIRIWindow());
        mWindow->SetEventCallback(EVENT_BIND_FUNCTION(KiriApplication::OnEvent));

        KiriRenderer::Init();

        mLayerImGui = new KiriLayerImGui();
        PushLayer(mLayerImGui);
    }

    KiriApplication::~KiriApplication() {}

    void KiriApplication::OnEvent(KiriEvent &e)
    {
        KiriEventDispatcher dispatcher(e);
        dispatcher.DispatchEvent<KiriWindowCloseEvent>(EVENT_BIND_FUNCTION(KiriApplication::OnWindowCloseEvent));
        dispatcher.DispatchEvent<KiriWindowResizeEvent>(EVENT_BIND_FUNCTION(KiriApplication::OnWindowResizeEvent));

        for (auto it = mLayerStack.end(); it != mLayerStack.begin();)
        {
            (*(--it))->OnEvent(e);
            if (e.mHandled)
                break;
        }
    }

    void KiriApplication::PushLayer(KiriLayer *Layer)
    {

        mLayerStack.PushLayer(Layer);
        Layer->OnAttach();
    }

    void KiriApplication::PopCurrentLayer()
    {

        mLayerStack.PopLayer(mExamplesList[mCurrentExampleName]);
    }

    void KiriApplication::PopLayer(KiriLayer *Layer)
    {
        mLayerStack.PopLayer(Layer);
    }

    void KiriApplication::Run()
    {
        int screen_shots_cnt = 0;
        mTimer.Restart();
        while (mRunning)
        {
            KiriTimeStep deltatime = (float)mTimer.Elapsed(true);

            // clear screen color
            // UInt height = GetWindow().GetWindowHeight();
            // UInt width = GetWindow().GetWindowWidth();
            // KIRI::KiriRendererCommand::SetViewport(Vector4F(0.f, 0.f, (float)width, (float)height));
            // KIRI::KiriRendererCommand::SetClearColor(Vector4F(0.1f, 0.1f, 0.1f, 1.f));
            // KIRI::KiriRendererCommand::Clear();

            if (!mMinimized)
                for (KiriLayer *Layer : mLayerStack)
                    Layer->OnUpdate(deltatime);

            mLayerImGui->begin();
            for (KiriLayer *Layer : mLayerStack)
                Layer->OnImguiRender();

            mLayerImGui->end();

            if (mCaptureScreen)
                KiriUtils::captureScreenshot(screen_shots_cnt++);

            // Render Logic
            mWindow->OnUpdate();
        }
    }

    void KiriApplication::ShutDown()
    {
        mRunning = false;
    }

    void KiriApplication::CaptureScreen(bool Enable)
    {
        mCaptureScreen = Enable;
    }

    bool KiriApplication::OnWindowCloseEvent(KiriWindowCloseEvent &e)
    {
        mRunning = false;
        return true;
    }

    bool KiriApplication::OnWindowResizeEvent(KiriWindowResizeEvent &e)
    {
        // When Window Minimized
        if (e.GetWindowHeight() == 0 || e.GetWindowWidth() == 0)
        {
            mMinimized = true;
            return false;
        }

        mMinimized = false;
        KiriRenderer::OnWindowResize(e.GetWindowWidth(), e.GetWindowHeight());

        return false;
    }
} // namespace KIRI
