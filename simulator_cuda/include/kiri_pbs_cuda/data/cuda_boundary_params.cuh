/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2021-02-11 01:26:00
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 15:44:07
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\data\cuda_boundary_params.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_BOUNDARY_PARAMS_CUH_
#define _CUDA_BOUNDARY_PARAMS_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{
    struct CudaBoundaryParams
    {
        float kernel_radius;
        float3 lowest_point;
        float3 highest_point;
        float3 world_size;
        float3 world_center;
        int3 grid_size;
    };

    extern CudaBoundaryParams CUDA_BOUNDARY_PARAMS;

} // namespace KIRI

#endif