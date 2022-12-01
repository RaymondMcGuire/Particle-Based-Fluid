/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2021-07-19 01:05:50
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-15 17:55:28
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sph\cuda_iisph_solver_common_gpu.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_IISPH_SOLVER_COMMON_GPU_CUH_
#define _CUDA_IISPH_SOLVER_COMMON_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{
    template <typename GradientFunc>
    __device__ void _ComputeDii(
        float3 *dii,
        const size_t i,
        const float3 *pos,
        const float *mass,
        const float densityi,
        size_t j,
        const size_t cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {
            if (i != j)
                *dii += -mass[j] / fmaxf(KIRI_EPSILON, densityi * densityi) * nablaW(pos[i] - pos[j]);
            ++j;
        }

        return;
    }

    template <typename GradientFunc>
    __device__ void _ComputeBoundaryDii(
        float3 *dii,
        const float3 posi,
        const float densityi,
        const float3 *bpos,
        const float *volume,
        const float rho0,
        size_t j,
        const size_t cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {
            *dii += -rho0 * volume[j] / fmaxf(KIRI_EPSILON, densityi * densityi) * nablaW(posi - bpos[j]);
            ++j;
        }
        return;
    }

    template <typename GradientFunc>
    __device__ void _ComputeDivergenceError(
        float *densityAdv,
        const size_t i,
        const float3 *pos,
        const float *mass,
        const float3 *vel,
        const float dt,
        size_t j,
        const size_t cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {
            if (i != j)
                *densityAdv += dt * mass[j] * dot((vel[i] - vel[j]), nablaW(pos[i] - pos[j]));
            ++j;
        }

        return;
    }

    template <typename GradientFunc>
    __device__ void _ComputeDivergenceErrorBoundary(
        float *densityAdv,
        const float3 posi,
        const float3 veli,
        const float dt,
        const float3 *bpos,
        const float *volume,
        const float rho0,
        size_t j,
        const size_t cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {
            *densityAdv += dt * rho0 * volume[j] * dot(veli, nablaW(posi - bpos[j]));
            ++j;
        }
        return;
    }

    template <typename GradientFunc>
    __device__ void _ComputeAii(
        float *aii,
        const size_t i,
        const float3 *pos,
        const float *mass,
        const float dpi,
        const float3 dii,
        size_t j,
        const size_t cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {
            if (i != j)
            {
                float3 kernel = nablaW(pos[i] - pos[j]);
                float3 dji = dpi * kernel;
                *aii += mass[j] * dot((dii - dji), kernel);
            }

            ++j;
        }

        return;
    }

    template <typename GradientFunc>
    __device__ void _ComputeBoundaryAii(
        float *aii,
        const float3 posi,
        const float dpi,
        const float3 dii,
        const float3 *bpos,
        const float *volume,
        const float rho0,
        size_t j,
        const size_t cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {
            float3 kernel = nablaW(posi - bpos[j]);
            float3 dji = dpi * kernel;
            *aii += rho0 * volume[j] * dot((dii - dji), kernel);
            ++j;
        }
        return;
    }

    template <typename GradientFunc>
    __device__ void _ComputeDijPj(
        float3 *dijpj,
        const size_t i,
        const float3 *pos,
        const float *mass,
        const float *density,
        const float *lastPressure,
        size_t j,
        const size_t cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {
            if (i != j)
            {
                float densityj2 = density[j] * density[j];
                *dijpj += -mass[j] / fmaxf(KIRI_EPSILON, densityj2) * lastPressure[j] * nablaW(pos[i] - pos[j]);
            }

            ++j;
        }

        return;
    }

    template <typename GradientFunc>
    __device__ void _ComputePressureSumParts(
        float *sum,
        const size_t i,
        const float3 *dijpj,
        const float3 *dii,
        const float *lastp,
        const float3 *pos,
        const float *mass,
        const float dpi,
        size_t j,
        const size_t cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {
            if (i != j)
            {
                float3 djkpk = dijpj[j];
                float3 kernel = nablaW(pos[i] - pos[j]);
                float3 dji = dpi * kernel;
                float3 dijpi = dji * lastp[i];

                float midotk = dot((dijpj[i] - dii[j] * lastp[j] - (djkpk - dijpi)), kernel);
                *sum += mass[j] * midotk;
            }

            ++j;
        }

        return;
    }

    template <typename GradientFunc>
    __device__ void _ComputeBoundaryPressureSumParts(
        float *sum,
        const float3 posi,
        const float3 dijpji,
        const float3 *bpos,
        const float *volume,
        const float rho0,
        size_t j,
        const size_t cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {
            *sum += rho0 * volume[j] * dot(dijpji, nablaW(posi - bpos[j]));
            ++j;
        }
        return;
    }

    template <typename GradientFunc>
    __device__ void _ComputePressureAcc(
        float3 *pacc,
        const size_t i,
        const float *pressure,
        const float3 *pos,
        const float *mass,
        const float *density,
        const float dpi,
        size_t j,
        const size_t cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {
            if (i != j)
            {
                float dpj = pressure[j] / fmaxf(KIRI_EPSILON, density[j] * density[j]);
                *pacc -= mass[j] * (dpi + dpj) * nablaW(pos[i] - pos[j]);
            }

            ++j;
        }

        return;
    }

    template <typename GradientFunc>
    __device__ void _ComputeBoundaryPressureAcc(
        float3 *pacc,
        const float3 posi,
        const float dpi,
        const float3 *bpos,
        const float *volume,
        const float rho0,
        size_t j,
        const size_t cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {
            *pacc -= rho0 * volume[j] * dpi * nablaW(posi - bpos[j]);
            ++j;
        }
        return;
    }

} // namespace KIRI

#endif /* _CUDA_IISPH_SOLVER_COMMON_GPU_CUH_ */