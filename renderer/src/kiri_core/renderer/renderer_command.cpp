/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-26 00:28:28
 * @LastEditTime: 2020-10-26 00:36:51
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\renderer\renderer_command.cpp
 */
#include <kiri_core/renderer/renderer_command.h>
#include <kiri_core/gui/opengl/opengl_renderer_api.h>

namespace KIRI
{
    KiriRendererAPI *KiriRendererCommand::sRendererAPI = new KiriOpenGLRendererAPI;
} // namespace KIRI
