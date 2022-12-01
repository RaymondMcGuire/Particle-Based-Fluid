/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2021-04-01 19:17:28
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-05-17 19:30:51
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\system\cuda_base_system_gpu.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_BASE_SYSTEM_GPU_CUH_
#define _CUDA_BASE_SYSTEM_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{

    static __global__ void TransferGPUData2VBO_CUDA(float4 *pos, float4 *col, float3 *lpos, float3 *lcol, const float radius, const size_t num, const size_t type)
    {
        const size_t i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num)
            return;

        pos[i] = make_float4(lpos[i], radius);
        col[i] = make_float4(lcol[i], type);
        return;
    }

    static __global__ void MRP1_GPU2VBO_CUDA(float4 *pos, float4 *col, float3 *lpos, float3 *lcol, float *radius, const size_t num, const size_t type)
    {
        const size_t i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num)
            return;

        pos[i] = make_float4(lpos[i], radius[i]);
        col[i] = make_float4(lcol[i], type);

        return;
    }

    static __global__ void TransferGPUData2VBO_CUDA(float4 *pos, float4 *col, float3 *lpos, float3 *lcol, float *radius, const size_t num, size_t *type)
    {
        const size_t i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num)
            return;

        pos[i] = make_float4(lpos[i], radius[i]);
        col[i] = make_float4(lcol[i], type[i]);
        return;
    }

} // namespace KIRI

#endif /* _CUDA_BASE_SYSTEM_GPU_CUH_ */