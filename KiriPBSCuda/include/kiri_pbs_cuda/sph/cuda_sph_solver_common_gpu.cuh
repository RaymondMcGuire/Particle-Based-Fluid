/*
 * @Author: Xu.WANG
 * @Date: 2020-07-04 14:48:23
 * @LastEditTime: 2021-02-15 15:14:21
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\sph\cuda_sph_solver_common_gpu.cuh
 */

#ifndef _CUDA_SPH_SOLVER_COMMON_GPU_CUH_
#define _CUDA_SPH_SOLVER_COMMON_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{

    static __global__ void BoundaryConstrain_CUDA(
        float3 *pos,
        float3 *vel,
        const uint num,
        const float3 lowestPoint,
        const float3 highestPoint,
        const float radius)
    {
        const uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num)
            return;

        float3 tmpPos = pos[i];
        float3 tmpVel = vel[i];

        if (tmpPos.x > highestPoint.x - 2 * radius)
        {
            tmpPos.x = highestPoint.x - 2 * radius;
            tmpVel.x = fminf(tmpVel.x, 0.0f);
            //tmpVel.x = 0.f;
        }

        if (tmpPos.x < lowestPoint.x + 2 * radius)
        {
            tmpPos.x = lowestPoint.x + 2 * radius;
            tmpVel.x = fmaxf(tmpVel.x, 0.0f);
            //tmpVel.x = 0.f;
        }

        if (tmpPos.y > highestPoint.y - 2 * radius)
        {
            tmpPos.y = highestPoint.y - 2 * radius;
            tmpVel.y = fminf(tmpVel.y, 0.0f);
            //tmpVel.y = 0.f;
        }

        if (tmpPos.y < lowestPoint.y + 2 * radius)
        {
            tmpPos.y = lowestPoint.y + 2 * radius;
            tmpVel.y = fmaxf(tmpVel.y, 0.0f);
            //tmpVel.y = 0.f;
        }

        if (tmpPos.z > highestPoint.z - 2 * radius)
        {
            tmpPos.z = highestPoint.z - 2 * radius;
            tmpVel.z = fminf(tmpVel.z, 0.0f);
            //tmpVel.z = 0.f;
        }

        if (tmpPos.z < lowestPoint.z + 2 * radius)
        {
            tmpPos.z = lowestPoint.z + 2 * radius;
            tmpVel.z = fmaxf(tmpVel.z, 0.0f);
            //tmpVel.z = 0.f;
        }

        pos[i] = tmpPos;
        vel[i] = tmpVel;

        return;
    }

    template <typename Func>
    __device__ void ComputeFluidDensity(
        float *density,
        const uint i,
        float3 *pos,
        float *mass,
        uint j,
        const uint cellEnd,
        Func W)
    {
        while (j < cellEnd)
        {
            *density += mass[j] * W(length(pos[i] - pos[j]));
            ++j;
        }

        return;
    }

    template <typename Func>
    __device__ void ComputeBoundaryDensity(
        float *density,
        const float3 posi,
        float3 *bpos,
        float *volume,
        const float rho0,
        uint j,
        const uint cellEnd,
        Func W)
    {
        while (j < cellEnd)
        {
            *density += rho0 * volume[j] * W(length(posi - bpos[j]));
            ++j;
        }
        return;
    }

    template <typename GradientFunc>
    __device__ void ComputeBoundaryPressure(
        float3 *a,
        const float3 posi,
        const float densityi,
        const float pressurei,
        const float3 *bpos,
        float *volume,
        const float rho0,
        uint j,
        const uint cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {
            *a += -rho0 * volume[j] * (pressurei / fmaxf(KIRI_EPSILON, densityi * densityi)) * nablaW(posi - bpos[j]);
            //*a += -volume[j] * (pressurei / rho0) * nablaW(posi - bpos[j]);
            ++j;
        }
        return;
    }

    template <typename GradientFunc>
    __device__ void ComputeBoundaryViscosity(
        float3 *a,
        const float3 posi,
        const float3 *bpos,
        const float3 veli,
        float densityi,
        float *volume,
        const float bnu,
        const float rho0,
        uint j,
        const uint cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {

            float3 dpij = posi - bpos[j];

            float dot_dvdp = dot(veli, dpij);
            if (dot_dvdp < 0.f)
            {
                float pij = -bnu / (2.f * densityi) * (dot_dvdp / (lengthSquared(dpij) + KIRI_EPSILON));
                *a += -volume[j] * rho0 * pij * nablaW(dpij);
            }

            ++j;
        }
        return;
    }

    template <typename GradientFunc>
    __device__ void ComputeFluidPressure(
        float3 *a,
        const uint i,
        float3 *pos,
        float *mass,
        float *density,
        float *pressure,
        uint j,
        const uint cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {
            if (i != j)
                *a += -mass[j] * (pressure[i] / fmaxf(KIRI_EPSILON, density[i] * density[i]) + pressure[j] / fmaxf(KIRI_EPSILON, density[j] * density[j])) * nablaW(pos[i] - pos[j]);
            ++j;
        }

        return;
    }

    template <typename LaplacianFunc>
    __device__ void ViscosityMuller2003(
        float3 *a,
        const uint i,
        float3 *pos,
        float3 *vel,
        float *mass,
        float *density,
        uint j,
        const uint cellEnd,
        LaplacianFunc nablaW2)
    {
        while (j < cellEnd)
        {
            *a += mass[j] * ((vel[j] - vel[i]) / density[j]) * nablaW2(length(pos[i] - pos[j]));
            ++j;
        }
        return;
    }

    template <typename GradientFunc>
    __device__ void ArtificialViscosity(
        float3 *a,
        const uint i,
        float3 *pos,
        float3 *vel,
        float *mass,
        float *density,
        const float nu,
        uint j,
        const uint cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {

            float3 dpij = pos[i] - pos[j];
            float3 dv = vel[i] - vel[j];

            float dot_dvdp = dot(dv, dpij);
            if (dot_dvdp < 0.f)
            {
                float pij = -nu / (density[i] + density[j]) * (dot_dvdp / (lengthSquared(dpij) + KIRI_EPSILON));
                *a += -mass[j] * pij * nablaW(dpij);
            }

            ++j;
        }
        return;
    }

    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func>
    __global__ void ComputeDensity_CUDA(
        float3 *pos,
        float *mass,
        float *density,
        const float rho0,
        const uint num,
        uint *cellStart,
        float3 *bPos,
        float *bVolume,
        uint *bCellStart,
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

            ComputeFluidDensity(&density[i], i, pos, mass, cellStart[hashIdx], cellStart[hashIdx + 1], W);
            ComputeBoundaryDensity(&density[i], pos[i], bPos, bVolume, rho0, bCellStart[hashIdx], bCellStart[hashIdx + 1], W);
        }

        return;
    }

    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename GradientFunc, typename LaplacianFunc>
    __global__ void ComputeViscosityTerm_CUDA(
        float3 *pos,
        float3 *vel,
        float3 *acc,
        float *mass,
        float *density,
        const float rho0,
        const float visc,
        const float bnu,
        const uint num,
        uint *cellStart,
        float3 *bPos,
        float *bVolume,
        uint *bCellStart,
        const int3 gridSize,
        Pos2GridXYZ p2xyz,
        GridXYZ2GridHash xyz2hash,
        GradientFunc nablaW,
        LaplacianFunc nablaW2)
    {
        const uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num)
            return;

        float3 a = make_float3(0.f);
        int3 gridXYZ = p2xyz(pos[i]);

#pragma unroll
        for (int m = 0; m < 27; ++m)
        {

            int3 curGridXYZ = gridXYZ + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
            const uint hashIdx = xyz2hash(curGridXYZ.x, curGridXYZ.y, curGridXYZ.z);
            if (hashIdx == (gridSize.x * gridSize.y * gridSize.z))
                continue;

            ViscosityMuller2003(&a, i, pos, vel, mass, density, cellStart[hashIdx], cellStart[hashIdx + 1], nablaW2);
            ComputeBoundaryViscosity(&a, pos[i], bPos, vel[i], density[i], bVolume, bnu, rho0, bCellStart[hashIdx], bCellStart[hashIdx + 1], nablaW);
        }

        acc[i] += visc * a;
        return;
    }

    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename GradientFunc>
    __global__ void ComputeArtificialViscosityTerm_CUDA(
        float3 *pos,
        float3 *vel,
        float3 *acc,
        float *mass,
        float *density,
        const float rho0,
        const float nu,
        const float bnu,
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

        float3 a = make_float3(0.0f);
        int3 gridXYZ = p2xyz(pos[i]);

#pragma unroll
        for (int m = 0; m < 27;
             ++m)
        {

            int3 curGridXYZ = gridXYZ + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
            const uint hashIdx = xyz2hash(curGridXYZ.x, curGridXYZ.y, curGridXYZ.z);
            if (hashIdx == (gridSize.x * gridSize.y * gridSize.z))
                continue;

            ArtificialViscosity(&a, i, pos, vel, mass, density, nu, cellStart[hashIdx], cellStart[hashIdx + 1], nablaW);
            ComputeBoundaryViscosity(&a, pos[i], bPos, vel[i], density[i], bVolume, bnu, rho0, bCellStart[hashIdx], bCellStart[hashIdx + 1], nablaW);
        }

        acc[i] += a;
        return;
    }

} // namespace KIRI

#endif /* _CUDA_SPH_SOLVER_COMMON_GPU_CUH_ */