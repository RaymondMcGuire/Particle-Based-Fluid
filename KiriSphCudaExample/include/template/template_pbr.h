/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-26 15:25:05
 * @LastEditTime: 2020-11-25 20:18:05
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriExamples\include\template\template_pbr.h
 */

#ifndef _KIRI_TEMPLATE_PBR_H_
#define _KIRI_TEMPLATE_PBR_H_

#include <KIRI.h>
namespace KIRI
{
    class KiriTemplatePBR : public KiriLayer
    {
    public:
        KiriTemplatePBR()
            : KiriLayer("KiriTemplatePBR"),
              mCamera(Vector3F(0.0f, 0.0f, 3.0f), Vector3F(0.0f, 0.0f, 1.0f), Vector3F(0.0f, 1.0f, 0.0f), 45.0f, 1280.f / 720.f),
              mScene(1280.f, 720.f), mFrameBuffer(1280.f, 720.f) {}

        KiriTemplatePBR(String Name, UInt WindowWidth, UInt WindowHeight)
            : KiriLayer(Name),
              mCamera(Vector3F(0.0f, 0.0f, 3.0f), Vector3F(0.0f, 0.0f, 1.0f), Vector3F(0.0f, 1.0f, 0.0f), 45.0f, (float)WindowWidth / (float)WindowHeight),
              mScene(WindowWidth, WindowHeight), mFrameBuffer(WindowWidth, WindowHeight), mWidth(WindowWidth), mHeight(WindowHeight) {}

    protected:
        virtual void OnAttach() override;
        virtual void OnEvent(KIRI::KiriEvent &e) override;

        virtual void OnImguiRender() = 0;

        virtual void SetupPBRParams() = 0;
        virtual void SetupPBRScene() = 0;

        KiriCameraFPC mCamera;
        KiriScene mScene;
        KiriFrameBuffer mFrameBuffer;

    private:
        void OnUpdate(const KIRI::KiriTimeStep &DeltaTime) override;
        UInt mWidth, mHeight;
    };
} // namespace KIRI
#endif