/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 12:05:59
 * @FilePath: \Kiri\core\src\kiri_core\gui\layer_imgui.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_core/gui/layer_imgui.h>

#include <imgui/include/imgui.h>
#include <imgui/include/imgui_impl_glfw.h>
#include <imgui/include/imgui_impl_opengl3.h>

#include <kiri_core/gui/opengl/opengl_window.h>

#include <kiri_application.h>
#include <kiri_params.h>

namespace KIRI
{
    KiriLayerImGui::KiriLayerImGui()
        : KiriLayer("KiriLayerImGui")
    {
    }

    KiriLayerImGui::~KiriLayerImGui()
    {
    }

    void KiriLayerImGui::OnAttach()
    {
        // ImGui Setup
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsLight();

        // Setup Platform/Renderer bindings
        GLFWwindow *window = (GLFWwindow *)KiriApplication::Get().GetWindow().GetNativeWindow();

        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 410");
    }

    void KiriLayerImGui::OnDetach()
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    void KiriLayerImGui::begin()
    {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void KiriLayerImGui::end()
    {
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    void KiriLayerImGui::ShowFPS(bool *Fps)
    {
        const float DISTANCE = 30.0f;
        static Int corner = 1;
        ImGuiIO &io = ImGui::GetIO();
        if (corner != -1)
        {
            ImVec2 window_pos = ImVec2((corner & 1) ? io.DisplaySize.x - DISTANCE : DISTANCE, (corner & 2) ? io.DisplaySize.y - DISTANCE : DISTANCE);
            ImVec2 window_pos_pivot = ImVec2((corner & 1) ? 1.0f : 0.0f, (corner & 2) ? 1.0f : 0.0f);
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
        }
        ImGui::SetNextWindowBgAlpha(0.35f); // Transparent background
        if (ImGui::Begin("Debug Tool", Fps, (corner != -1 ? ImGuiWindowFlags_NoMove : 0) | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
        {
            ImGui::Text("Average %.3f ms/frame (%.1f FPS)", 1000.0 / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        }
        ImGui::End();
    }

    void KiriLayerImGui::ShowSceneInfo(bool *SceneInfo)
    {
        if (ImGui::Begin("Scene Panel", SceneInfo))
        {
            ImGui::Text("Camera Setting");
            ImGui::Checkbox("Camera Debug Mode", &CAMERA_PARAMS.debug);
        }
        ImGui::End();
    }

    void KiriLayerImGui::SwitchApp(String AppName)
    {
        auto &app = KiriApplication::Get();
        app.PopCurrentLayer();
        app.SetCurrentExampleName(AppName);
        app.PushLayer(app.CurrentExampleName());
    }

    void KiriLayerImGui::OnImguiRender()
    {

        KiriApplication::Get().CaptureScreen(mScreenShot);

        if (!mScreenShot)
        {
            if (ImGui::BeginMainMenuBar())
            {
                if (ImGui::BeginMenu("Files"))
                {
                    if (ImGui::MenuItem("Quit", "Alt+F4"))
                    {
                        KiriApplication::Get().ShutDown();
                    }
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Panels"))
                {

                    ImGui::Checkbox("FPS", &mFps);
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Tools"))
                {

                    ImGui::Checkbox("Screen Capture", &mScreenShot);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Examples"))
                {

                    if (ImGui::MenuItem("SphApp"))
                    {
                        SwitchApp("sph_app");
                    }

                    if (ImGui::MenuItem("MultiSphRen14App"))
                    {
                        SwitchApp("multisph_ren14_app");
                    }

                    if (ImGui::MenuItem("MultiSphYang15App"))
                    {
                        SwitchApp("multisph_yang15_app");
                    }

                    ImGui::EndMenu();
                }

                ImGui::EndMainMenuBar();
            }

            if (mFps)
            {
                ShowFPS(&mFps);
            }

            if (mSceneInfo)
            {
                ShowSceneInfo(&mSceneInfo);
            }
        }
        else
        {
            // ScreenShot Mode -> Change to FullScreen
            if (!KiriApplication::Get().GetWindow().IsFullscreen())
            {
                const GLFWvidmode *mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
                auto window = static_cast<GLFWwindow *>(KiriApplication::Get().GetWindow().GetNativeWindow());
                glfwSetWindowMonitor(window, glfwGetPrimaryMonitor(), 0, 0, mode->width, mode->height, mode->refreshRate);
            }
        }

        // static bool show = true;
        // ImGui::ShowDemoWindow(&show);
        // KIRI_LOG_DEBUG("Is full screen mode={0}", KiriApplication::Get().GetWindow().IsFullscreen());
    }
} // namespace KIRI