/*** 
 * @Author: Xu.WANG
 * @Date: 2021-02-10 15:29:35
 * @LastEditTime: 2021-02-14 00:01:36
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\data\cuda_sph_params.h
 */

#ifndef _CUDA_SPH_PARAMS_H_
#define _CUDA_SPH_PARAMS_H_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{
    struct CudaSphParams
    {

        float rest_mass;
        float rest_density;
        float particle_radius;
        float kernel_radius;

        bool atf_visc;
        float stiff;
        float visc;
        float nu;
        float bnu;

        float3 gravity;

        float dt;
    };

    struct CudaSphAppParams
    {
        bool run = false;
        bool run_offline = false;

        int scene_data_idx = 0;
        char bgeo_file_name[32] = "default";
        bool bgeo_export = false;
    };

    extern CudaSphParams CUDA_SPH_PARAMS;
    extern CudaSphAppParams CUDA_SPH_APP_PARAMS;

} // namespace KIRI

#endif