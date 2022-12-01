/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2021-03-02 20:30:53
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-15 17:52:36
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sph\cuda_akinci13_gpu.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_AKINCI13_GPU_CUH_
#define _CUDA_AKINCI13_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{

    template <typename GradientFunc>
    __device__ void _ComputeFluidNormal(
        float3 *normal,
        const size_t i,
        float3 *pos,
        float *mass,
        float *density,
        const float h,
        size_t j,
        const size_t cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {
            if (i != j)
                *normal += mass[j] / density[j] * nablaW(pos[i] - pos[j]);
            ++j;
        }
        *normal *= h;
        return;
    }

    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename GradientFunc>
    __global__ void _ComputeNormal_CUDA(
        float3 *pos,
        float3 *normal,
        float *mass,
        float *density,
        const float h,
        const size_t num,
        size_t *cellStart,
        const int3 gridSize,
        Pos2GridXYZ p2xyz,
        GridXYZ2GridHash xyz2hash,
        GradientFunc nablaW)
    {
        const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num)
            return;

        int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
        for (int m = 0; m < 27; ++m)
        {
            int3 cur_grid_xyz = grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
            const size_t hash_idx = xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
            if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
                continue;

            _ComputeFluidNormal(&normal[i], i, pos, mass, density, h, cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
        }

        return;
    }

    template <typename CohesionFunc>
    __device__ void _ComputeFluidSurfaceTension(
        float3 *a,
        float3 *pos,
        float *mass,
        float *density,
        float3 *normal,
        const float rho0,
        const float gamma,
        const size_t i,
        size_t j,
        const size_t cellEnd,
        CohesionFunc C)
    {
        while (j < cellEnd)
        {
            if (i != j)
            {
                float3 dir = pos[i] - pos[j];
                float dist = length(dir);
                float3 curvatureTerm = normal[i] - normal[j];
                float3 cohesionTerm = mass[j] * C(dist) * dir / fmaxf(KIRI_EPSILON, dist);
                float kij = 2.f * rho0 / fmaxf(KIRI_EPSILON, (density[i] + density[j]));
                *a += gamma * kij * (curvatureTerm + cohesionTerm);
            }
            ++j;
        }
        return;
    }

    template <typename AdhesionFunc>
    __device__ void _ComputeBoundaryAdhesionTerm(
        float3 *a,
        const float3 posi,
        const float3 *bpos,
        float *volume,
        const float rho0,
        const float beta,
        size_t j,
        const size_t cellEnd,
        AdhesionFunc A)
    {
        while (j < cellEnd)
        {
            float3 dir = posi - bpos[j];
            float dist = length(dir);
            *a += beta * rho0 * volume[j] * A(dist) * dir / fmaxf(KIRI_EPSILON, dist);
            ++j;
        }
        return;
    }

    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename CohesionFunc, typename AdhesionFunc>
    __global__ void _ComputeAkinci13Term_CUDA(
        float3 *pos,
        float3 *acc,
        float *mass,
        float *density,
        float3 *normal,
        const float rho0,
        const float gamma,
        const float beta,
        const size_t num,
        size_t *cellStart,
        float3 *bPos,
        float *bVolume,
        size_t *bCellStart,
        const int3 gridSize,
        Pos2GridXYZ p2xyz,
        GridXYZ2GridHash xyz2hash,
        CohesionFunc C,
        AdhesionFunc A)
    {
        const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num)
            return;

        float3 a = make_float3(0.f);
        int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
        for (int m = 0; m < 27; ++m)
        {

            int3 cur_grid_xyz = grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
            const size_t hash_idx = xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
            if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
                continue;

            _ComputeFluidSurfaceTension(&a, pos, mass, density, normal, rho0, gamma, i, cellStart[hash_idx], cellStart[hash_idx + 1], C);
            _ComputeBoundaryAdhesionTerm(&a, pos[i], bPos, bVolume, rho0, beta, bCellStart[hash_idx], bCellStart[hash_idx + 1], A);
        }

        acc[i] += -a;
        return;
    }

} // namespace KIRI

#endif /* _CUDA_SPH_SOLVER_GPU_CUH_ */