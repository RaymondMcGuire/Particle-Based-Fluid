/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-11 11:44:38
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-15 17:57:10
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sph\cuda_multi_sph_solver_gpu.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_MULTI_SPH_SOLVER_GPU_CUH_
#define _CUDA_MULTI_SPH_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{

  __global__ void _ComputeMRPressure_CUDA(float *density, float *pressure,
                                          const size_t num, const float rho0,
                                          const float stiff)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    pressure[i] = stiff * (density[i] - rho0);

    return;
  }

  __global__ void _MRBoundaryConstrain_CUDA(float3 *pos, float3 *vel, float *rad,
                                            const size_t num,
                                            const float3 lowestPoint,
                                            const float3 highestPoint)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    float3 tmp_pos = pos[i];
    float3 tmp_vel = vel[i];
    float radius = rad[i];

    if (tmp_pos.x > highestPoint.x - 2 * radius)
    {
      tmp_pos.x = highestPoint.x - 2 * radius;
      tmp_vel.x = fminf(tmp_vel.x, 0.f);
    }

    if (tmp_pos.x < lowestPoint.x + 2 * radius)
    {
      tmp_pos.x = lowestPoint.x + 2 * radius;
      tmp_vel.x = fmaxf(tmp_vel.x, 0.f);
    }

    if (tmp_pos.y > highestPoint.y - 2 * radius)
    {
      tmp_pos.y = highestPoint.y - 2 * radius;
      tmp_vel.y = fminf(tmp_vel.y, 0.f);
    }

    if (tmp_pos.y < lowestPoint.y + 2 * radius)
    {
      tmp_pos.y = lowestPoint.y + 2 * radius;
      tmp_vel.y = fmaxf(tmp_vel.y, 0.f);
    }

    if (tmp_pos.z > highestPoint.z - 2 * radius)
    {
      tmp_pos.z = highestPoint.z - 2 * radius;
      tmp_vel.z = fminf(tmp_vel.z, 0.f);
    }

    if (tmp_pos.z < lowestPoint.z + 2 * radius)
    {
      tmp_pos.z = lowestPoint.z + 2 * radius;
      tmp_vel.z = fmaxf(tmp_vel.z, 0.f);
    }

    tmp_vel = make_float3(0.f);

    pos[i] = tmp_pos;
    vel[i] = tmp_vel;

    return;
  }

  template <typename Func>
  __device__ void _ComputeMRFluidDensity(float *density, const size_t i,
                                         const float3 *pos, const float *mass,
                                         const float kernelRadiusI, size_t j,
                                         const size_t cellEnd, Func W)
  {
    while (j < cellEnd)
    {
      // float pij = length(pos[i] - pos[j]);
      // // printf("pij=%.3f, kernelRadiusI=%.3f \n", pij, kernelRadiusI);

      float pij = length(pos[i] - pos[j]);
      if (pij < kernelRadiusI)
        *density += mass[j] * W(pij);
      // printf("mass=%.3f, W(pij)=%.3f \n", mass[j], W(pij));

      ++j;
    }

    return;
  }

  template <typename Func>
  __device__ void
  _ComputeMRBoundaryDensity(float *density, const float3 posi, const float3 *bpos,
                            const float *volume, const float kernelRadiusI,
                            const float rho0, size_t j, const size_t cellEnd,
                            Func W)
  {
    while (j < cellEnd)
    {
      float pij = length(posi - bpos[j]);
      if (pij < kernelRadiusI)
        *density += rho0 * volume[j] * W(pij);
      ++j;
    }
    return;
  }

  template <typename GradientFunc>
  __device__ void _ComputeMRBoundaryPressure(
      float3 *a, const float3 posi, const float densityi, const float pressurei,
      const float3 *bpos, float *volume, const float kernelRadiusI,
      const float rho0, size_t j, const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      float pij = length(posi - bpos[j]);
      if (pij < kernelRadiusI)
        *a += -rho0 * volume[j] *
              (pressurei / fmaxf(KIRI_EPSILON, densityi * densityi)) *
              nablaW(posi - bpos[j]);
      ++j;
    }
    return;
  }

  template <typename GradientFunc>
  __device__ void
  _ComputeMRBoundaryViscosity(float3 *a, const float3 posi, const float3 *bpos,
                              const float3 veli, const float densityi,
                              const float *volume, const float kernelRadiusI,
                              const float bnu, const float rho0, size_t j,
                              const size_t cellEnd, GradientFunc nablaW)
  {
    auto h2 = kernelRadiusI * kernelRadiusI;
    while (j < cellEnd)
    {
      float pij = length(posi - bpos[j]);
      if (pij < kernelRadiusI)
      {
        float3 dpij = posi - bpos[j];
        float dot_dvdp = dot(veli, dpij);
        float pij = 10.f * bnu / densityi *
                    (dot_dvdp / (lengthSquared(dpij) + 0.01f * h2));
        *a += volume[j] * rho0 * pij * nablaW(dpij);
      }

      ++j;
    }
    return;
  }

  template <typename GradientFunc>
  __device__ void
  _ComputeMRFluidPressure(float3 *a, const size_t i, float3 *pos, float *mass,
                          float *density, float *pressure,
                          const float kernelRadiusI, size_t j,
                          const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      float pij = length(pos[i] - pos[j]);
      if (pij < kernelRadiusI)
        if (i != j)
          *a -= mass[j] *
                (pressure[i] / fmaxf(KIRI_EPSILON, density[i] * density[i]) +
                 pressure[j] / fmaxf(KIRI_EPSILON, density[j] * density[j])) *
                nablaW(pos[i] - pos[j]);
      ++j;
    }

    return;
  }

  template <typename LaplacianFunc>
  __device__ void
  _MRViscosityMuller2003(float3 *a, const size_t i, float3 *pos, float3 *vel,
                         float *mass, float *density, const float kernelRadiusI,
                         size_t j, const size_t cellEnd, LaplacianFunc nablaW2)
  {
    while (j < cellEnd)
    {
      float pij = length(pos[i] - pos[j]);
      if (pij < kernelRadiusI)
        if (i != j)
          *a += mass[j] * ((vel[j] - vel[i]) / fmaxf(KIRI_EPSILON, density[j])) *
                nablaW2(length(pos[i] - pos[j]));
      ++j;
    }
    return;
  }

  template <typename GradientFunc>
  __device__ void _MRArtificialViscosity(float3 *a, const size_t i, float3 *pos,
                                         float3 *vel, float *mass, float *density,
                                         const float kernelRadiusI, const float nu,
                                         size_t j, const size_t cellEnd,
                                         GradientFunc nablaW)
  {
    auto h2 = kernelRadiusI * kernelRadiusI;
    while (j < cellEnd)
    {

      float pij = length(pos[i] - pos[j]);
      if (pij < kernelRadiusI)
      {
        if (i != j)
        {
          float3 dpij = pos[i] - pos[j];
          float3 dv = vel[i] - vel[j];

          float dot_dvdp = dot(dv, dpij);

          float pij = 10.f * nu / density[j] *
                      (dot_dvdp / (lengthSquared(dpij) + 0.01f * h2));
          *a += mass[j] * pij * nablaW(dpij);
        }
      }

      ++j;
    }
    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash>
  __global__ void
  _ComputeMRDensity_CUDA(float3 *pos, float *mass, float *density, float *radius,
                         const float rho0, const size_t num, size_t *cellStart,
                         float3 *bPos, float *bVolume, size_t *bCellStart,
                         const int3 gridSize, Pos2GridXYZ p2xyz,
                         GridXYZ2GridHash xyz2hash)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    int3 grid_xyz = p2xyz(pos[i]);
    density[i] = 0.f;

#pragma unroll
    for (int m = 0; m < 27; ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputeMRFluidDensity(&density[i], i, pos, mass, radius[i] * 5.f,
                             cellStart[hash_idx], cellStart[hash_idx + 1],
                             Poly6Kernel(radius[i] * 5.f));
      _ComputeMRBoundaryDensity(&density[i], pos[i], bPos, bVolume,
                                radius[i] * 5.f, rho0, bCellStart[hash_idx],
                                bCellStart[hash_idx + 1],
                                Poly6Kernel(radius[i] * 5.f));
    }
    // printf("density[i]=%.3f \n", density[i]);

    if (density[i] != density[i])
      printf("sph density nan!! density[i]=%.3f \n", density[i]);

    if (density[i] == 0.f)
      printf("sph density = 0.0!! density[i]=%.3f \n", density[i]);

    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash>
  __global__ void
  _ComputeMRViscosityTerm_CUDA(float3 *pos, float3 *vel, float3 *acc, float *mass,
                               float *density, float *radius, const float rho0,
                               const float visc, const float bnu, const size_t num,
                               size_t *cellStart, float3 *bPos, float *bVolume,
                               size_t *bCellStart, const int3 gridSize,
                               Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    float3 a = make_float3(0.f);
    int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
    for (int m = 0; m < 27; ++m)
    {

      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _MRViscosityMuller2003(&a, i, pos, vel, mass, density, radius[i] * 5.f,
                             cellStart[hash_idx], cellStart[hash_idx + 1],
                             ViscosityKernelLaplacian(radius[i] * 5.f));
    }

    if (a.x != a.x || a.y != a.y || a.z != a.z)
    {
      printf("_ComputeMuller03Viscosity acc nan!! a=%.3f,%.3f,%.3f \n",
             KIRI_EXPAND_FLOAT3(a));
    }

    acc[i] += visc * a;
    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash>
  __global__ void _ComputeMRArtificialViscosityTerm_CUDA(
      float3 *pos, float3 *vel, float3 *acc, float *mass, float *density,
      float *radius, const float rho0, const float nu, const float bnu,
      const size_t num, size_t *cellStart, float3 *bPos, float *bVolume,
      size_t *bCellStart, const int3 gridSize, Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    float3 a = make_float3(0.f);
    int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
    for (int m = 0; m < 27; ++m)
    {

      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _MRArtificialViscosity(&a, i, pos, vel, mass, density, radius[i] * 5.f, nu,
                             cellStart[hash_idx], cellStart[hash_idx + 1],
                             SpikyKernelGrad(radius[i] * 5.f));
      _ComputeMRBoundaryViscosity(&a, pos[i], bPos, vel[i], density[i], bVolume,
                                  radius[i] * 5.f, bnu, rho0, bCellStart[hash_idx],
                                  bCellStart[hash_idx + 1],
                                  SpikyKernelGrad(radius[i] * 5.f));
    }

    if (a.x != a.x || a.y != a.y || a.z != a.z)
    {
      printf("_ComputeArtificialViscosity acc nan!! a=%.3f,%.3f,%.3f \n",
             KIRI_EXPAND_FLOAT3(a));
    }

    acc[i] += a;
    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash>
  __global__ void
  _ComputeMRNablaTerm_CUDA(float3 *pos, float3 *acc, float *mass, float *density,
                           float *pressure, float *radius, const float rho0,
                           const size_t num, size_t *cellStart, float3 *bPos,
                           float *bVolume, size_t *bCellStart, const int3 gridSize,
                           Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    auto a = make_float3(0.f);
    int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
    for (int m = 0; m < 27; ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputeMRFluidPressure(&a, i, pos, mass, density, pressure, radius[i] * 5.f,
                              cellStart[hash_idx], cellStart[hash_idx + 1],
                              SpikyKernelGrad(radius[i] * 5.f));
      _ComputeMRBoundaryPressure(&a, pos[i], density[i], pressure[i], bPos,
                                 bVolume, radius[i] * 5.f, rho0,
                                 bCellStart[hash_idx], bCellStart[hash_idx + 1],
                                 SpikyKernelGrad(radius[i] * 5.f));
    }
    if (a.x != a.x || a.y != a.y || a.z != a.z)
    {
      printf("Nabla acc nan!! a=%.3f,%.3f,%.3f \n", KIRI_EXPAND_FLOAT3(a));
    }
    acc[i] += a;
    return;
  }

} // namespace KIRI

#endif /* _CUDA_MULTI_SPH_SOLVER_GPU_CUH_ */