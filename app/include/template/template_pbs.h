/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-05-11 15:17:58
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-06-24 16:01:27
 * @FilePath: \Kiri\KiriExamples\include\template\template_pbs.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _KIRI_TEMPLATE_PBS_H_
#define _KIRI_TEMPLATE_PBS_H_

#include <KIRI.h>
#include <kiri_cuda_utils.h>
#include <kiri_utils.h>

namespace KIRI {
class KiriTemplatePBS : public KiriLayer {
public:
  KiriTemplatePBS() : KiriLayer("KiriTemplatePBS"), mFrameBuffer(1280, 720) {
    mCamera = std::make_shared<KiriCameraFPC>(
        CameraProperty(Vector3F(0.0f, 0.0f, 3.0f), Vector3F(0.0f, 0.0f, 1.0f),
                       Vector3F(0.0f, 1.0f, 0.0f), 45.0f, 1280.f / 720.f));
    mScene = std::make_shared<KiriScene>(1280, 720);
  }

  KiriTemplatePBS(String Name, UInt WindowWidth, UInt WindowHeight)
      : KiriLayer(Name), mFrameBuffer(WindowWidth, WindowHeight),
        mWidth(WindowWidth), mHeight(WindowHeight) {
    mCamera = std::make_shared<KiriCameraFPC>(
        CameraProperty(Vector3F(0.0f, 0.0f, 3.0f), Vector3F(0.0f, 0.0f, 1.0f),
                       Vector3F(0.0f, 1.0f, 0.0f), 45.0f,
                       (float)WindowWidth / (float)WindowHeight));
    mScene = std::make_shared<KiriScene>(WindowWidth, WindowHeight);
    mFluidRenderSystem = std::make_shared<KiriFluidRenderSystem>(
        WindowWidth, WindowHeight, mCamera);
    mParticleRenderSystem = std::make_shared<KiriParticleRenderSystem>(mCamera);

    KIRI_LOG_DEBUG("Loaded FlatBuffer Binary Data:{0}", Name);
    mSceneConfigData = ImportBinaryFile(Name);
  }

  void Clear() {
    mCamera = std::make_shared<KiriCameraFPC>(CameraProperty(
        Vector3F(0.0f, 0.0f, 3.0f), Vector3F(0.0f, 0.0f, 1.0f),
        Vector3F(0.0f, 1.0f, 0.0f), 45.0f, (float)mWidth / (float)mHeight));
    mScene = std::make_shared<KiriScene>(mWidth, mHeight);
    mFluidRenderSystem =
        std::make_shared<KiriFluidRenderSystem>(mWidth, mHeight, mCamera);
    mParticleRenderSystem = std::make_shared<KiriParticleRenderSystem>(mCamera);
  }

  virtual ~KiriTemplatePBS() noexcept {};

protected:
  virtual void OnAttach() override;
  virtual void OnDetach() override;
  virtual void OnEvent(KIRI::KiriEvent &e) override;

  virtual void OnImguiRender() = 0;
  virtual void SetupPBSParams() = 0;
  virtual void SetupPBSScene() = 0;
  virtual void OnPBSUpdate(const KIRI::KiriTimeStep &DeltaTime){};

  void ChangeSceneConfigData(String Name);

  void SetDebugParticlesWithRadius(Array1<Vector4F> particles);
  void SetParticleWithRadius(Array1Vec4F pos, Array1Vec4F col, UInt num);
  void SetParticleVBOWithRadius(UInt PosVBO, UInt ColorVBO, UInt Number);

  Vec_Char mSceneConfigData;
  KiriCameraFPCPtr mCamera;
  KiriScenePtr mScene;
  KiriFluidRenderSystemPtr mFluidRenderSystem;
  KiriParticleRenderSystemPtr mParticleRenderSystem;

  UInt mSimCount = 0;

  float mRenderInterval;
  void SetRenderFps(float Fps);

private:
  void OnUpdate(const KIRI::KiriTimeStep &DeltaTime) override;
  Vec_Char ImportBinaryFile(String const &Name);

  KiriFrameBuffer mFrameBuffer;
  UInt mWidth, mHeight;
};
} // namespace KIRI
#endif