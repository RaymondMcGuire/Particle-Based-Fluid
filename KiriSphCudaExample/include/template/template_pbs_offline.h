/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-26 15:25:05
 * @LastEditTime: 2020-12-04 23:40:38
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriExamples\include\template\template_pbs_offline.h
 */

#ifndef _KIRI_TEMPLATE_PBS_OFFLINE_H_
#define _KIRI_TEMPLATE_PBS_OFFLINE_H_

#include <KIRI.h>
namespace KIRI
{
    class KiriTemplatePBSOffline : public KiriLayer
    {
    public:
        KiriTemplatePBSOffline()
            : KiriLayer("KiriTemplatePBSOffline") {}

        KiriTemplatePBSOffline(String Name, UInt WindowWidth, UInt WindowHeight)
            : KiriLayer(Name), mWidth(WindowWidth), mHeight(WindowHeight)
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
        UInt mSimCount = 0;

    private:
        void OnUpdate(const KIRI::KiriTimeStep &DeltaTime) override;
        Vec_Char ImportBinaryFile(String const &Name);

        UInt mWidth, mHeight;
    };
} // namespace KIRI
#endif