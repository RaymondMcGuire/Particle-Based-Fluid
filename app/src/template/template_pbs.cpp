/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:27
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 15:19:07
 * @FilePath: \Kiri\app\src\template\template_pbs.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <template/template_pbs.h>
namespace KIRI
{

  void KiriTemplatePBS::SetRenderFps(float Fps)
  {
    mRenderInterval = 1.f / Fps;
    KIRI_LOG_INFO("Delta t = {0}, FPS = {1}", mRenderInterval, Fps);
  }

  void KiriTemplatePBS::ChangeSceneConfigData(String Name)
  {
    if (mName == Name)
      return;

    mName = Name;
    this->OnDetach();
    mSceneConfigData = ImportBinaryFile(mName);
    this->OnAttach();
  }

  void KiriTemplatePBS::SetDebugParticlesWithRadius(Array1<Vector4F> particles)
  {
    mParticleRenderSystem->SetParticles(particles);
  }

  void KiriTemplatePBS::SetParticleVBOWithRadius(UInt PosVBO, UInt ColorVBO,
                                                 UInt Number)
  {
    mFluidRenderSystem->SetParticlesVBOWithRadius(PosVBO, ColorVBO, Number);
  }

  void KiriTemplatePBS::SetParticleWithRadius(Array1Vec4F pos, Array1Vec4F col,
                                              UInt num)
  {
    mFluidRenderSystem->SetParticlesWithRadius(pos, col, num);
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
    mScene->EnableCubeSkybox(true, "pool_2k");
    mFluidRenderSystem->SetSkyBoxTex(mScene->GetCubeSkybox()->GetEnvCubeMap());

    SetupPBSParams();
    SetupPBSScene();
  }

  void KiriTemplatePBS::OnDetach() { this->Clear(); }

  void KiriTemplatePBS::OnUpdate(const KIRI::KiriTimeStep &DeltaTime)
  {
    // KIRI_LOG_TRACE("Delta Time {0}, ms: {1}, fps: {2}", DeltaTime.seconds(),
    // DeltaTime.milliSeconds(), DeltaTime.fps());
    OnPBSUpdate(DeltaTime);

    mCamera->OnUpdate(DeltaTime);
    KIRI::KiriRenderer::BeginScene(mCamera);

    mFrameBuffer.Bind();
    mFrameBuffer.EnableDepthTest();

    // render scene
    // mScene->renderShadow();
    KIRI::KiriRendererCommand::SetViewport(
        Vector4F(0.f, 0.f, (float)mWidth, (float)mHeight));
    KIRI::KiriRendererCommand::SetClearColor(Vector4F(0.1f, 0.1f, 0.1f, 1.f));
    KIRI::KiriRendererCommand::Clear();
    mScene->RenderCubeSkybox();
    mScene->Render();
    mParticleRenderSystem->RenderParticles();
    mFrameBuffer.DisableDepthTest();
    mFrameBuffer.Release();

    KIRI::KiriRendererCommand::SetViewport(
        Vector4F(0.f, 0.f, (float)mWidth, (float)mHeight));
    KIRI::KiriRendererCommand::SetClearColor(Vector4F(0.1f, 0.1f, 0.1f, 1.f));
    KIRI::KiriRendererCommand::Clear();
    mFluidRenderSystem->renderFluid(mFrameBuffer.TextureColorBuffer(),
                                    mFrameBuffer.TextureDepthBuffer());

    KIRI::KiriRenderer::EndScene();
  }

  void KiriTemplatePBS::OnEvent(KIRI::KiriEvent &e)
  {
    // KIRI_LOG_TRACE("{0}", e);
  }
} // namespace KIRI