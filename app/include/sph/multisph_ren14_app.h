/***
 * @Author: Xu.WANG
 * @Date: 2020-10-26 15:25:05
 * @LastEditTime: 2021-04-20 14:05:21
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriExamples\include\sph\multisph_app.h
 */

#ifndef _KIRI_MULTISPH_REN14_APP_H_
#define _KIRI_MULTISPH_REN14_APP_H_

#include <template/template_pbs.h>
#include <kiri_pbs_cuda/system/cuda_multisph_ren14_system.cuh>

namespace KIRI
{
    class KiriMultiSphRen14App final : public KiriTemplatePBS
    {
    public:
        KiriMultiSphRen14App()
            : KiriTemplatePBS() {}
        KiriMultiSphRen14App(String Name, Int WindowWidth, Int WindowHeight)
            : KiriTemplatePBS(Name, WindowWidth, WindowHeight) {}

    protected:
        virtual void OnImguiRender() override;
        virtual void SetupPBSParams() override;
        virtual void SetupPBSScene() override;
        virtual void OnPBSUpdate(const KIRI::KiriTimeStep &DeltaTime) override;

    private:
        CudaMultiSphRen14SystemPtr mSystem;
    };
} // namespace KIRI
#endif