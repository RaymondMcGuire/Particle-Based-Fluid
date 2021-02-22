/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-26 17:26:05
 * @LastEditTime: 2021-02-22 11:32:57
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \SPH_CUDA\KiriSphCudaExample\src\template\template_pbs.cpp
 */
#include <template/template_pbs.h>
namespace KIRI
{

    void KiriTemplatePBS::ChangeSceneConfigData(String Name)
    {
        if (mName == Name)
            return;

        mName = Name;
        mSceneConfigData = ImportBinaryFile(mName);
        mScene->Clear();
        auto &app = KiriApplication::Get();
        app.PopCurrentLayer();
        app.PushLayer(app.ExamplesList()[app.CurrentExampleName()]);
    }

    void KiriTemplatePBS::SetParticleVBOWithRadius(UInt PosVBO, UInt ColorVBO, UInt Number)
    {
        mFluidRenderSystem->SetParticlesVBOWithRadius(PosVBO, ColorVBO, Number);
    }

    Vec_Char KiriTemplatePBS::ImportBinaryFile(String const &Name)
    {
        String importPath = String(DB_PBR_PATH) + "sceneconfig/" + Name + ".bin";

        if (RELEASE && PUBLISH)
        {
            importPath = "./resources/sceneconfig/" + Name + ".bin";
        }

        std::ifstream importer(importPath, std::ios::binary);
        KIRI_LOG_INFO("Import Scene Conifg File From:{0}", importPath);

        return Vec_Char(std::istreambuf_iterator<char>(importer),
                        std::istreambuf_iterator<char>());
    }

    void KiriTemplatePBS::OnAttach()
    {
        mScene->enableCubeSkybox(true, "pool_2k");
        mFluidRenderSystem->SetSkyBoxTex(mScene->getCubeSkybox()->getEnvCubeMap());

        SetupPBSParams();
        SetupPBSScene();
    }

    void KiriTemplatePBS::OnUpdate(const KIRI::KiriTimeStep &DeltaTime)
    {
        //KIRI_LOG_TRACE("Delta Time {0}, ms: {1}, fps: {2}", DeltaTime.GetSeconds(), DeltaTime.GetMilliSeconds(), DeltaTime.GetFps());
        OnPBSUpdate(DeltaTime);

        mCamera->OnUpdate(DeltaTime);
        KIRI::KiriRenderer::BeginScene(mCamera);

        mFrameBuffer.Bind();
        mFrameBuffer.EnableDepthTest();

        // render scene
        //mScene->renderShadow();
        KIRI::KiriRendererCommand::SetViewport(Vector4F(0.f, 0.f, (float)mWidth, (float)mHeight));
        KIRI::KiriRendererCommand::SetClearColor(Vector4F(0.1f, 0.1f, 0.1f, 1.f));
        KIRI::KiriRendererCommand::Clear();
        mScene->renderCubeSkybox();
        mScene->render();

        mFrameBuffer.DisableDepthTest();
        mFrameBuffer.Release();

        KIRI::KiriRendererCommand::SetViewport(Vector4F(0.f, 0.f, (float)mWidth, (float)mHeight));
        KIRI::KiriRendererCommand::SetClearColor(Vector4F(0.1f, 0.1f, 0.1f, 1.f));
        KIRI::KiriRendererCommand::Clear();
        mFluidRenderSystem->renderFluid(mFrameBuffer.TextureColorBuffer(), mFrameBuffer.TextureDepthBuffer());

        KIRI::KiriRenderer::EndScene();
    }

    void KiriTemplatePBS::OnEvent(KIRI::KiriEvent &e)
    {
        //KIRI_LOG_TRACE("{0}", e);
    }
} // namespace KIRI