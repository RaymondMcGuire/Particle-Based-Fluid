/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-26 17:26:05
 * @LastEditTime: 2020-11-25 21:56:59
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriExamples\src\template\template_pbr.cpp
 */
#include <template/template_pbr.h>

namespace KIRI
{
    void KiriTemplatePBR::OnAttach()
    {
        mScene.enableCubeSkybox(true, "pool_2k");
        SetupPBRParams();
        SetupPBRScene();
    }
    void KiriTemplatePBR::OnUpdate(const KIRI::KiriTimeStep &DeltaTime)
    {
        mCamera.OnUpdate(DeltaTime);

        KIRI::KiriRenderer::BeginScene(mCamera);

        mFrameBuffer.Bind();
        mFrameBuffer.EnableDepthTest();
        // render scene
        mScene.renderShadow();
        KIRI::KiriRendererCommand::SetViewport(Vector4F(0.f, 0.f, (float)mWidth, (float)mHeight));
        KIRI::KiriRendererCommand::SetClearColor(Vector4F(0.1f, 0.1f, 0.1f, 1.f));
        KIRI::KiriRendererCommand::Clear();
        mScene.renderCubeSkybox();
        mScene.render();
        mFrameBuffer.DisableDepthTest();
        mFrameBuffer.Release();

        mFrameBuffer.RenderToScreen();

        KIRI::KiriRenderer::EndScene();
    }

    void KiriTemplatePBR::OnEvent(KIRI::KiriEvent &e)
    {
        //KIRI_LOG_TRACE("{0}", e);
    }
} // namespace KIRI