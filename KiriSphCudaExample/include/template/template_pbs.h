/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-26 15:25:05
 * @LastEditTime: 2020-11-24 02:52:12
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
              mCamera(Vector3F(0.0f, 0.0f, 3.0f), Vector3F(0.0f, 0.0f, 1.0f), Vector3F(0.0f, 1.0f, 0.0f), 45.0f, 1280.f / 720.f),
              mScene(1280.f, 720.f), mFrameBuffer(1280.f, 720.f), mFluidRenderSystem(1280.f, 720.f) {}

        KiriTemplatePBS(String Name, UInt WindowWidth, UInt WindowHeight)
            : KiriLayer(Name),
              mCamera(Vector3F(0.0f, 0.0f, 3.0f), Vector3F(0.0f, 0.0f, 1.0f), Vector3F(0.0f, 1.0f, 0.0f), 45.0f, (float)WindowWidth / (float)WindowHeight),
              mScene(WindowWidth, WindowHeight), mFrameBuffer(WindowWidth, WindowHeight), mFluidRenderSystem(WindowWidth, WindowHeight),
              mWidth(WindowWidth), mHeight(WindowHeight)
        {
            mSceneConfigData = ImportBinaryFile(Name);
        }

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
        KiriCameraFPC mCamera;
        KiriScene mScene;
        KiriFluidRenderSystem mFluidRenderSystem;

        UInt mSimCount = 0;

    private:
        void OnUpdate(const KIRI::KiriTimeStep &DeltaTime) override;
        Vec_Char ImportBinaryFile(String const &Name);

        KiriFrameBuffer mFrameBuffer;
        UInt mWidth, mHeight;
    };
} // namespace KIRI
#endif