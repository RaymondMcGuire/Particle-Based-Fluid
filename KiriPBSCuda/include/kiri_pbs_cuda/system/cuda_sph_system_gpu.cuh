/*
 * @Author: Xu.WANG
 * @Date: 2020-07-04 14:48:23
 * @LastEditTime: 2021-02-10 00:22:16
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\system\cuda_sph_system_gpu.cuh
 */

#ifndef _CUDA_SPH_SYSTEM_GPU_CUH_
#define _CUDA_SPH_SYSTEM_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{

    __global__ void CopyGPUData2VBO_CUDA(float4 *pos, float4 *col, float3 *lpos, float3 *lcol, const uint num, const float radius)
    {
        const uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num)
            return;

        pos[i] = make_float4(lpos[i], radius);
        col[i] = make_float4(lcol[i], 0.f);
        return;
    }

    template <typename Func>
    __device__ void ComputeBoundaryVolume(
        float *delta,
        const uint i,
        float3 *pos,
        uint j,
        const uint cellEnd,
        Func W)
    {
        while (j < cellEnd)
        {
            *delta += W(length(pos[i] - pos[j]));
            ++j;
        }
        return;
    }

    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func>
    __global__ void ComputeBoundaryVolume_CUDA(
        float3 *pos,
        float *volume,
        const uint num,
        uint *cellStart,
        const int3 gridSize,
        Pos2GridXYZ p2xyz,
        GridXYZ2GridHash xyz2hash,
        Func W)
    {
        const uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num)
            return;

        int3 gridXYZ = p2xyz(pos[i]);

#pragma unroll
        for (int m = 0; m < 27; ++m)
        {
            int3 curGridXYZ = gridXYZ + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
            const uint hashIdx = xyz2hash(curGridXYZ.x, curGridXYZ.y, curGridXYZ.z);
            if (hashIdx == (gridSize.x * gridSize.y * gridSize.z))
                continue;

            ComputeBoundaryVolume(&volume[i], i, pos, cellStart[hashIdx], cellStart[hashIdx + 1], W);
        }

        volume[i] = 1.f / fmaxf(volume[i], KIRI_EPSILON);
        return;
    }

} // namespace KIRI

#endif /* _CUDA_SPH_SYSTEM_GPU_CUH_ */