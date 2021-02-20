/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-25 12:25:41
 * @LastEditTime: 2020-10-26 02:00:28
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\renderer\renderer_command.h
 */

#ifndef _KIRI_RENDERER_COMMAND_H_
#define _KIRI_RENDERER_COMMAND_H_
#pragma once
#include <kiri_core/renderer/renderer_api.h>

namespace KIRI
{
    class KiriRendererCommand
    {
    public:
        inline static void Init()
        {
            sRendererAPI->Init();
        }

        inline static void Clear(bool depth=true)
        {
            sRendererAPI->Clear(depth);
        }

        inline static void SetViewport(const Vector4F &rect)
        {
            sRendererAPI->SetViewport(rect);
        }

        inline static void SetClearColor(const Vector4F &color)
        {
            sRendererAPI->SetClearColor(color);
        }

        inline static void GlobalUboGenerate()
        {
            sRendererAPI->GlobalUboGenerate();
        }

        inline static void GlobalUboBind(const KiriCameraPtr&camera)
        {
            sRendererAPI->GlobalUboBind(camera);
        }

    private:
        static KiriRendererAPI *sRendererAPI;
    };
} // namespace KIRI

#endif