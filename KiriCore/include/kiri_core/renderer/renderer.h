/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 12:25:41
 * @LastEditTime: 2020-11-03 22:17:30
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\renderer\renderer.h
 */

#ifndef _KIRI_RENDERER_H_
#define _KIRI_RENDERER_H_
#pragma once
#include <kiri_core/renderer/renderer_command.h>

namespace KIRI
{
    class KiriRenderer
    {
    public:
        static void Init();
        static void OnWindowResize(UInt width, UInt height);
        static void BeginScene(const KiriCameraPtr &camera);
        static void EndScene();

        inline static KiriRendererAPI::RenderPlatform GetRenderPlatform() { return KiriRendererAPI::GetRenderPlatform(); }

    private:
        struct SceneData
        {
        };

        static SceneData *mSceneData;
    };
} // namespace KIRI

#endif