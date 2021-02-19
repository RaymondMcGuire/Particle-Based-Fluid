/*** 
 * @Author: Xu.WANG
 * @Date: 2021-02-10 15:56:02
 * @LastEditTime: 2021-02-10 16:02:20
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\data\cuda_boundary_params.h
 */

#ifndef _CUDA_BOUNDARY_PARAMS_H_
#define _CUDA_BOUNDARY_PARAMS_H_

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