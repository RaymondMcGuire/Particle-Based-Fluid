/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-26 15:25:05
 * @LastEditTime: 2021-02-20 01:38:49
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriExamples\include\template\template_pbs.h
 */

#ifndef _KIRI_TEMPLATE_PBS_H_
#define _KIRI_TEMPLATE_PBS_H_

#include <KIRI.h>
namespace KIRI
{
    class KiriTemplatePBS : public KiriLayer
    {
    public:
        KiriTemplatePBS()
            : KiriLayer("KiriTemplatePBS"),
              mFrameBuffer(1280, 720)
        {
            mCamera = std::make_shared<KiriCameraFPC>(CameraProperty(Vector3F(0.0f, 0.0f, 3.0f), Vector3F(0.0f, 0.0f, 1.0f), Vector3F(0.0f, 1.0f, 0.0f), 45.0f, 1280.f / 720.f));
            mScene = std::make_shared<KiriScene>(1280, 720, mCamera);
        }

        KiriTemplatePBS(String Name, UInt WindowWidth, UInt WindowHeight)
            : KiriLayer(Name),
              mFrameBuffer(WindowWidth, WindowHeight),
              mWidth(WindowWidth),
              mHeight(WindowHeight)
        {
            mCamera = std::make_shared<KiriCameraFPC>(CameraProperty(Vector3F(0.0f, 0.0f, 3.0f), Vector3F(0.0f, 0.0f, 1.0f), Vector3F(0.0f, 1.0f, 0.0f), 45.0f, (float)WindowWidth / (float)WindowHeight));
            mScene = std::make_shared<KiriScene>(WindowWidth, WindowHeight, mCamera);
            mFluidRenderSystem = std::make_shared<KiriFluidRenderSystem>(WindowWidth, WindowHeight, mCamera);

            mSceneConfigData = ImportBinaryFile(Name);
        }

        virtual ~KiriTemplatePBS() noexcept {};

    protected:
        virtual void OnAttach() override;
        virtual void OnEvent(KIRI::KiriEvent &e) override;

        virtual void OnImguiRender() = 0;
        virtual void SetupPBSParams() = 0;
        virtual void SetupPBSScene() = 0;
        virtual void OnPBSUpdate(const KIRI::KiriTimeStep &DeltaTime){};

        void ChangeSceneConfigData(String Name);
        void SetParticleVBOWithRadius(UInt PosVBO, UInt ColorVBO, UInt Number);

        Vec_Char mSceneConfigData;
        KiriCameraFPCPtr mCamera;
        KiriScenePtr mScene;
        KiriFluidRenderSystemPtr mFluidRenderSystem;

        UInt mSimCount = 0;

    private:
        void OnUpdate(const KIRI::KiriTimeStep &DeltaTime) override;
        Vec_Char ImportBinaryFile(String const &Name);

        KiriFrameBuffer mFrameBuffer;
        UInt mWidth, mHeight;
    };
} // namespace KIRI
#endif