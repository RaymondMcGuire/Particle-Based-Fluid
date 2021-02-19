/*
 * @Author: Xu.WANG
 * @Date: 2020-07-04 14:48:23
 * @LastEditTime: 2021-02-14 21:24:00
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\sph\cuda_sph_solver_gpu.cuh
 */

#ifndef _CUDA_SPH_SOLVER_GPU_CUH_
#define _CUDA_SPH_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{

    __global__ void ComputePressure_CUDA(
        float *density,
        float *pressure,
        const uint num,
        const float rho0,
        const float stiff)
    {
        const uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num)
            return;

        pressure[i] = stiff * (density[i] - rho0);

        return;
    }

    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename GradientFunc>
    __global__ void ComputeNablaTerm_CUDA(
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

        acc[i] += a;
        return;
    }

} // namespace KIRI

#endif /* _CUDA_SPH_SOLVER_GPU_CUH_ */