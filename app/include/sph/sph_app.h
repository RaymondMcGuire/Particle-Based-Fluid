/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2021-02-11 01:51:54
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 16:14:35
 * @FilePath: \Kiri\KiriExamples\include\sph\sph_app.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
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
            : KiriTemplatePBS() {}
        KiriSphApp(String Name, Int WindowWidth, Int WindowHeight)
            : KiriTemplatePBS(Name, WindowWidth, WindowHeight) {}

    protected:
        virtual void OnImguiRender() override;
        virtual void SetupPBSParams() override;
        virtual void SetupPBSScene() override;
        virtual void OnPBSUpdate(const KIRI::KiriTimeStep &DeltaTime) override;

    private:
        CudaSphSystemPtr mSystem;

        KiriBoxPtr mBoundaryModel;
        KiriEntityPtr mBoundaryEnity;

        float3 mInitLowestPoint, mInitHighestPoint;
    };
} // namespace KIRI
#endif