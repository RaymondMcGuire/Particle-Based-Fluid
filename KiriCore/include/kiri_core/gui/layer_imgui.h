/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 03:19:55
 * @LastEditTime: 2021-02-20 19:45:45
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\gui\layer_imgui.h
 */
#ifndef _KIRI_LAYER_IMGUI_H_
#define _KIRI_LAYER_IMGUI_H_
#pragma once
#include <kiri_core/gui/layer.h>

namespace KIRI
{
    class KiriLayerImGui : public KiriLayer
    {
    public:
        KiriLayerImGui();
        ~KiriLayerImGui();

        virtual void OnAttach() override;
        virtual void OnDetach() override;
        virtual void OnImguiRender() override;

        void begin();
        void end();

    protected:
        void ShowFPS(bool *Fps);
        void ShowSceneInfo(bool *SceneInfo);

    private:
        bool mScreenShot = false;
        bool mFps = true;
        bool mSceneInfo = false;

        void SwitchApp(String AppName);
    };
} // namespace KIRI
#endif