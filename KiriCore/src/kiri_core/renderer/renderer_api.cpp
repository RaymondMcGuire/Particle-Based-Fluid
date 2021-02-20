/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-26 00:24:59
 * @LastEditTime: 2020-11-02 11:03:28
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\renderer\renderer_api.cpp
 */

#include <kiri_core/renderer/renderer_api.h>

namespace KIRI
{
    KiriRendererAPI::RenderPlatform KiriRendererAPI::sRenderPlatform = KiriRendererAPI::RenderPlatform::OpenGL;
} // namespace KIRI
