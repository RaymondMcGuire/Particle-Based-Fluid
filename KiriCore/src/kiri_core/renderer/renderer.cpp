/*** 
 * @Author: Jayden Zhang
 * @Date: 2020-10-26 00:45:12
 * @LastEditTime: 2020-11-03 22:18:05
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\renderer\renderer.cpp
 */
#include <kiri_core/renderer/renderer.h>
namespace KIRI
{
    KiriRenderer::SceneData *KiriRenderer::mSceneData = new KiriRenderer::SceneData;

    void KiriRenderer::Init()
    {
        KiriRendererCommand::Init();
        KiriRendererCommand::GlobalUboGenerate();
    }

    void KiriRenderer::OnWindowResize(UInt width, UInt height)
    {
        KiriRendererCommand::SetViewport(Vector4F(0.f, 0.f, (float)width, (float)height));
    }

    void KiriRenderer::BeginScene(const KiriCameraPtr&camera)
    {
        KiriRendererCommand::GlobalUboBind(camera);
    }

    void KiriRenderer::EndScene()
    {
    }

} // namespace KIRI
