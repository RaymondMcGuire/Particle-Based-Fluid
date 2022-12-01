/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-13 17:15:50
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-21 19:09:51
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sph\cuda_pbf_solver_common_gpu.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_PBF_SOLVER_COMMON_GPU_CUH_
#define _CUDA_PBF_SOLVER_COMMON_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{

  template <typename GradientFunc>
  __device__ void _ComputeVorticityOmega(
      float3 *omega,
      const size_t i,
      const float3 *pos,
      const float3 *vel,
      const float *mass,
      const float *density,
      const size_t cellStart,
      const size_t cellEnd,
      GradientFunc nablaW)
  {
    size_t j = cellStart;
    while (j < cellEnd)
    {
      if (i != j)
        *omega -= mass[j] / density[j] * cross(vel[i] - vel[j], nablaW(pos[i] - pos[j]));

      ++j;
    }

    return;
  }

  template <typename GradientFunc>
  __device__ void _ComputeVorticityConfinement(
      float3 *eta,
      const size_t i,
      const float3 *normOmega,
      const float3 *pos,
      const float *mass,
      const float *density,
      const size_t cellStart,
      const size_t cellEnd,
      GradientFunc nablaW)
  {
    size_t j = cellStart;
    while (j < cellEnd)
    {
      if (i != j)
        *eta += mass[j] / density[j] * normOmega[j] * nablaW(pos[i] - pos[j]);

      ++j;
    }

    return;
  }

  template <typename GradientFunc>
  __device__ void _ComputePBFGradC(
      float *sumGradC2,
      float3 *gradCI,
      const size_t i,
      const float rho0,
      const float3 *pos,
      const float *mass,
      const size_t cellStart,
      const size_t cellEnd,
      GradientFunc nablaW)
  {
    size_t j = cellStart;
    while (j < cellEnd)
    {
      if (i != j)
      {
        float3 grad_cj = -mass[j] / rho0 * nablaW(pos[i] - pos[j]);
        *sumGradC2 += lengthSquared(grad_cj);
        *gradCI += -grad_cj;
      }

      ++j;
    }

    return;
  }

  template <typename GradientFunc>
  __device__ void
  _ComputePBFBoundaryGradC(
      float3 *gradCI,
      const float3 posI,
      const float rho0,
      const float3 *posB,
      const float *volumeB,
      const size_t cellStart,
      const size_t cellEnd,
      GradientFunc nablaW)
  {
    size_t j = cellStart;
    while (j < cellEnd)
    {
      *gradCI += rho0 * volumeB[j] * nablaW(posI - posB[j]);
      ++j;
    }

    return;
  }

  template <typename GradientFunc>
  __device__ void _ComputePBFDeltaPosIncompressible(
      float3 *delta,
      const size_t i,
      const float rho0,
      const float *lambda,
      const float3 *pos,
      const float *mass,
      const size_t cellStart,
      const size_t cellEnd,
      GradientFunc nablaW)
  {
    size_t j = cellStart;
    while (j < cellEnd)
    {
      float3 dp = pos[i] - pos[j];
      float3 grad_cj = mass[j] / rho0 * nablaW(dp);
      *delta += (lambda[i] + lambda[j]) * grad_cj;
      ++j;
    }

    return;
  }

  template <typename Func, typename GradientFunc>
  __device__ void _ComputePBFDeltaPosRealtime(
      float3 *delta,
      const size_t i,
      const float rho0,
      const float *lambda,
      const float3 *pos,
      const float *mass,
      const float deltaQ,
      const float corrK,
      const float corrN,
      const float dt,
      const size_t cellStart,
      const size_t cellEnd,
      Func W,
      GradientFunc nablaW)
  {
    size_t j = cellStart;
    while (j < cellEnd)
    {
      if (i != j)
      {
        float3 dp = pos[i] - pos[j];
        float corr = -corrK * powf(W(length(dp)) / W(deltaQ), corrN) * dt;
        float3 grad_cj = mass[i] / rho0 * nablaW(dp);
        *delta += (lambda[i] + lambda[j] + corr) * grad_cj;
      }

      ++j;
    }

    return;
  }

  template <typename GradientFunc>
  __device__ void _ComputePBFBoundaryDeltaPosIncompressible(
      float3 *delta,
      const float3 posI,
      const float lambdaI,
      const float3 *posB,
      const float *volumeB,
      const size_t cellStart,
      const size_t cellEnd,
      GradientFunc nablaW)
  {
    size_t j = cellStart;
    while (j < cellEnd)
    {
      float3 dp = posI - posB[j];
      *delta += lambdaI * volumeB[j] * nablaW(dp);
      ++j;
    }

    return;
  }
} // namespace KIRI

#endif /* _CUDA_PBF_SOLVER_COMMON_GPU_CUH_ */