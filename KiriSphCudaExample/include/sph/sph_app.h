/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-26 15:25:05
 * @LastEditTime: 2021-02-14 21:41:18
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriExamples\include\sph\sph_app.h
 */

#ifndef _KIRI_SPH_APP_H_
#define _KIRI_SPH_APP_H_

#include <template/template_pbs.h>
#include <kiri_pbs_cuda/system/cuda_sph_system.cuh>

namespace KIRI
{
    class KiriSphApp final : public KiriTemplatePBS
    {
    public:
        KiriSphApp()
            : KiriTemplatePBS(), mSimRepeatNumer(1) {}
        KiriSphApp(String Name, Int WindowWidth, Int WindowHeight)
            : KiriTemplatePBS(Name, WindowWidth, WindowHeight), mSimRepeatNumer(1) {}

    protected:
        virtual void OnImguiRender() override;
        virtual void SetupPBSParams() override;
        virtual void SetupPBSScene() override;
        virtual void OnPBSUpdate(const KIRI::KiriTimeStep &DeltaTime) override;

    private:
        void SetRenderFps(float Fps);
        Int mSimRepeatNumer;
        CudaSphSystemPtr mSystem;
    };
} // namespace KIRI
#endif