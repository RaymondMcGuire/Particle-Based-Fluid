/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-13 20:37:06
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 17:47:43
 * @FilePath: \Kiri\KiriExamples\include\sph\multisph_yang15_app.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _KIRI_MULTISPH_YANG15_APP_H_
#define _KIRI_MULTISPH_YANG15_APP_H_

#include <template/template_pbs.h>
#include <kiri_pbs_cuda/system/cuda_multisph_yang15_system.cuh>

namespace KIRI
{
    class KiriMultiSphYang15App final : public KiriTemplatePBS
    {
    public:
        KiriMultiSphYang15App()
            : KiriTemplatePBS() {}
        KiriMultiSphYang15App(String Name, Int WindowWidth, Int WindowHeight)
            : KiriTemplatePBS(Name, WindowWidth, WindowHeight) {}

    protected:
        virtual void OnImguiRender() override;
        virtual void SetupPBSParams() override;
        virtual void SetupPBSScene() override;
        virtual void OnPBSUpdate(const KIRI::KiriTimeStep &DeltaTime) override;

    private:
        CudaMultiSphYang15SystemPtr mSystem;

        KiriBoxPtr mBoundaryModel;
        KiriEntityPtr mBoundaryEnity;

        float3 mInitLowestPoint, mInitHighestPoint;
    };
} // namespace KIRI
#endif