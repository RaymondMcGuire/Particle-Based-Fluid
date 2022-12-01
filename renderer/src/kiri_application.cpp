/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 13:39:09
 * @FilePath: \core\src\kiri_application.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_application.h>
#include <kiri_core/renderer/renderer.h>
#include <root_directory.h>

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

    mLayerImGui = std::make_shared<KiriLayerImGui>();
    mLayerStack.PushLayer(mLayerImGui);
    mLayerImGui->OnAttach();
  }

  KiriApplication::~KiriApplication() {}

  void KiriApplication::OnEvent(KiriEvent &e)
  {
    KiriEventDispatcher dispatcher(e);
    dispatcher.DispatchEvent<KiriWindowCloseEvent>(
        EVENT_BIND_FUNCTION(KiriApplication::OnWindowCloseEvent));
    dispatcher.DispatchEvent<KiriWindowResizeEvent>(
        EVENT_BIND_FUNCTION(KiriApplication::OnWindowResizeEvent));

    for (auto it = mLayerStack.end(); it != mLayerStack.begin();)
    {
      (*(--it))->OnEvent(e);
      if (e.mHandled)
        break;
    }
  }

  void KiriApplication::PushLayer(const String &layerName)
  {

    mLayerStack.PushOverlay(mExamplesList[layerName]);
    mExamplesList[layerName]->OnAttach();
  }

  void KiriApplication::PopCurrentLayer()
  {
    mLayerStack.PopOverlay(mExamplesList[mCurrentExampleName]);
  }

  void KiriApplication::PopLayer(const String &layerName)
  {
    mLayerStack.PopOverlay(mExamplesList[layerName]);
  }

  const char *CreateBaseNameForVideo(Int cnt, const char *ext,
                                     const char *prefix)
  {
    static char basename[30];

    snprintf(basename, sizeof(basename), "%s%04d.%s", prefix, cnt, ext);

    return basename;
  }

#include <stb_image.h>
#include <stb_image_write.h>

#define TINYOBJLOADER_IMPLEMENTATION

  void FlipVertically(Int mWidth, Int height, char *data)
  {
    char rgb[3];

    for (Int y = 0; y < height / 2; ++y)
    {
      for (Int x = 0; x < mWidth; ++x)
      {
        Int top = (x + y * mWidth) * 3;
        Int bottom = (x + (height - y - 1) * mWidth) * 3;

        memcpy(rgb, data + top, sizeof(rgb));
        memcpy(data + top, data + bottom, sizeof(rgb));
        memcpy(data + bottom, rgb, sizeof(rgb));
      }
    }
  }

  Int SaveScreenShot(const char *filename)
  {
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    Int x = viewport[0];
    Int y = viewport[1];
    Int mWidth = viewport[2];
    Int height = viewport[3];

    char *data =
        (char *)malloc((size_t)(mWidth * height * 3)); // 3 components (R, G, B)

    if (!data)
      return 0;

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(x, y, mWidth, height, GL_RGB, GL_UNSIGNED_BYTE, data);

    FlipVertically(mWidth, height, data);

    Int saved = stbi_write_png(filename, mWidth, height, 3, data, 0);

    free(data);

    return saved;
  }
  Int CaptureScreenShot(Int cnt)
  {
    char buildRootPath[200];
    strcpy_s(buildRootPath, 200, ROOT_PATH);
    strcat_s(buildRootPath, sizeof(buildRootPath), "/export/screenshots/");
    strcat_s(buildRootPath, sizeof(buildRootPath),
             CreateBaseNameForVideo(cnt, "png", ""));

    Int saved = SaveScreenShot(buildRootPath);

    if (saved)
      KIRI_LOG_INFO("Successfully Saved Image:{0}", buildRootPath);
    else
      KIRI_LOG_ERROR("Failed Saving Image:{0}", buildRootPath);

    return saved;
  }

  void KiriApplication::Run()
  {
    int screen_shots_cnt = 0;
    mTimer.Restart();
    while (bRunning)
    {
      KiriTimeStep deltaTime = (float)mTimer.Elapsed(true);

      // clear screen color
      // UInt height = GetWindow().GetWindowHeight();
      // UInt mWidth = GetWindow().GetWindowWidth();
      // KIRI::KiriRendererCommand::SetViewport(Vector4F(0.f, 0.f, (float)mWidth,
      // (float)height)); KIRI::KiriRendererCommand::SetClearColor(Vector4F(0.1f,
      // 0.1f, 0.1f, 1.f)); KIRI::KiriRendererCommand::Clear();

      if (!bMinimized)
        for (auto it = mLayerStack.begin(); it != mLayerStack.end(); it++)
          (*it)->OnUpdate(deltaTime);

      mLayerImGui->begin();
      for (auto it = mLayerStack.begin(); it != mLayerStack.end(); it++)
        (*it)->OnImguiRender();
      mLayerImGui->end();

      if (bCaptureScreen && !bMinimized)
        CaptureScreenShot(screen_shots_cnt++);

      // Render Logic
      mWindow->OnUpdate();
    }
  }

  void KiriApplication::ShutDown() { bRunning = false; }

  void KiriApplication::CaptureScreen(bool Enable) { bCaptureScreen = Enable; }

  bool KiriApplication::OnWindowCloseEvent(KiriWindowCloseEvent &e)
  {
    bRunning = false;
    return true;
  }

  bool KiriApplication::OnWindowResizeEvent(KiriWindowResizeEvent &e)
  {
    // When Window Minimized
    if (e.GetWindowHeight() == 0 || e.GetWindowWidth() == 0)
    {
      bMinimized = true;
      return false;
    }

    bMinimized = false;
    KiriRenderer::OnWindowResize(e.GetWindowWidth(), e.GetWindowHeight());

    return false;
  }
} // namespace KIRI
