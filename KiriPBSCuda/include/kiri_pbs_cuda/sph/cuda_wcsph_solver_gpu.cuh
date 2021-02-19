/*
 * @Author: Xu.WANG
 * @Date: 2020-07-04 14:48:23
 * @LastEditTime: 2021-02-16 19:31:35
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\sph\cuda_wcsph_solver_gpu.cuh
 */

#ifndef _CUDA_WCSPH_SOLVER_GPU_CUH_
#define _CUDA_WCSPH_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{
    __global__ void ComputePressureByTait_CUDA(
        float *density,
        float *pressure,
        const uint num,
        const float rho0,
        const float stiff,
        const float negativeScale)
    {
        const uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num)
            return;

        pressure[i] = stiff * (powf((density[i] / rho0), 7.f) - 1.0f);
        //float eos_scale = 100.f * 100.f * rho0;
        //pressure[i] = eos_scale / 7.f * (powf((density[i] / rho0), 7.f) - 1.0f);

        if (pressure[i] < 0.0f)
            pressure[i] *= negativeScale;

        return;
    }

    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename GradientFunc>
    __global__ void ComputeNablaTermConstrain_CUDA(
        float3 *pos,
        float3 *acc,
        float *mass,
        float *density,
        float *pressure,
        const float rho0,
        const uint num,
        uint *cellStart,
        float3 *bPos,
        float *bVolume,
        uint *bCellStart,
        const int3 gridSize,
        Pos2GridXYZ p2xyz,
        GridXYZ2GridHash xyz2hash,
        GradientFunc nablaW)
    {
        const uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num)
            return;

        auto a = make_float3(0.0f);
        int3 gridXYZ = p2xyz(pos[i]);

#pragma unroll
        for (int m = 0; m < 27; ++m)
        {
            int3 curGridXYZ = gridXYZ + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
            const uint hashIdx = xyz2hash(curGridXYZ.x, curGridXYZ.y, curGridXYZ.z);
            if (hashIdx == (gridSize.x * gridSize.y * gridSize.z))
                continue;

            ComputeFluidPressure(&a, i, pos, mass, density, pressure, cellStart[hashIdx], cellStart[hashIdx + 1], nablaW);
            ComputeBoundaryPressure(&a, pos[i], density[i], pressure[i], bPos, bVolume, rho0, bCellStart[hashIdx], bCellStart[hashIdx + 1], nablaW);
        }

        // if (length(a) > 1000.f)
        //     a = normalize(a) * 1000.f;

        acc[i] += a;
        return;
    }

} // namespace KIRI

#endif /* _CUDA_WCSPH_SOLVER_GPU_CUH_ */