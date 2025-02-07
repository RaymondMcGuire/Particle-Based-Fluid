/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-27 10:50:17
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-27 12:36:02
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sph\cuda_sph_solver_common_gpu.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_SPH_SOLVER_COMMON_GPU_CUH_
#define _CUDA_SPH_SOLVER_COMMON_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {
static __global__ void _ComputePressure_CUDA(float *pressure,
                                             const float *density,
                                             const size_t num, const float rho0,
                                             const float stiff,
                                             const float negativeScale) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  pressure[i] = stiff * (density[i] - rho0);

  if (pressure[i] < 0.f)
    pressure[i] *= negativeScale;

  return;
}

static __global__ void _ComputeVelMag_CUDA(float *velMag, const float3 *vel,
                                           const float3 *acc, const float dt,
                                           const size_t num) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  velMag[i] = lengthSquared(vel[i] + acc[i] * dt);

  return;
}

template <typename Func>
__device__ void _ComputeViscosityXSPH(float3 *a, const size_t i,
                                      const float3 *pos, const float3 *vel,
                                      const float *mass, const float *density,
                                      const float visc, const size_t cellStart,
                                      const size_t cellEnd, Func W) {
  size_t j = cellStart;
  while (j < cellEnd) {
    *a += -visc * 2.f * mass[j] / (density[i] + density[j]) *
          (vel[i] - vel[j]) * W(length(pos[i] - pos[j]));
    ++j;
  }
  return;
}

template <typename Func>
__device__ void
_ComputeBoundaryViscosityXSPH(float3 *a, const float3 posI, const float3 velI,
                              const float densityI, const float3 *posB,
                              const float *volumeB, const float rho0,
                              const float boundaryVisc, const size_t cellStart,
                              const size_t cellEnd, Func W) {

  size_t j = cellStart;
  while (j < cellEnd) {
    *a += -boundaryVisc * rho0 * volumeB[j] / densityI *
          (velI - make_float3(0.f)) * W(length(posI - posB[j]));
    ++j;
  }
  return;
}

template <typename Func>
__device__ void _ComputeFluidDensity(float *density, const size_t i,
                                     const float3 *pos, const float *mass,
                                     const size_t cellStart,
                                     const size_t cellEnd, Func W) {
  size_t j = cellStart;
  while (j < cellEnd) {
    if (i != j)
      *density += mass[j] * W(length(pos[i] - pos[j]));
    ++j;
  }

  return;
}

template <typename Func>
__device__ void
_ComputeBoundaryDensity(float *density, const float3 posI, const float3 *posB,
                        const float *volumeB, const float rho0,
                        const size_t cellStart, const size_t cellEnd, Func W) {
  size_t j = cellStart;
  while (j < cellEnd) {
    *density += rho0 * volumeB[j] * W(length(posI - posB[j]));
    ++j;
  }
  return;
}

template <typename GradientFunc>
__device__ void _computeBoundaryPressure(
    float3 *a, const float3 posI, const float densityI, const float pressureI,
    const float3 *posB, const float *volumeB, const float rho0,
    const size_t cellStart, const size_t cellEnd, GradientFunc nablaW) {
  size_t j = cellStart;
  while (j < cellEnd) {
    *a += -rho0 * volumeB[j] *
          (pressureI / fmaxf(KIRI_EPSILON, densityI * densityI)) *
          nablaW(posI - posB[j]);
    ++j;
  }
  return;
}

template <typename GradientFunc>
__device__ void _ComputeBoundaryArtificialViscosity(
    float3 *a, const float3 posI, const float3 *posB, const float3 velI,
    const float densityI, const float *volumeB, const float bnu,
    const float kernelRadius, const float rho0, const size_t cellStart,
    const size_t cellEnd, GradientFunc nablaW) {
  auto h2 = kernelRadius * kernelRadius;
  size_t j = cellStart;
  while (j < cellEnd) {

    float3 dpij = posI - posB[j];

    float dot_dvdp = dot(velI, dpij);

    float pij =
        10.f * bnu / densityI * (dot_dvdp / (lengthSquared(dpij) + 0.01f * h2));
    *a += volumeB[j] * rho0 * pij * nablaW(dpij);

    ++j;
  }
  return;
}

template <typename GradientFunc>
__device__ void
_ComputeFluidPressure(float3 *a, const size_t i, float3 *pos, const float *mass,
                      const float *density, const float *pressure,
                      const size_t cellStart, const size_t cellEnd,
                      GradientFunc nablaW) {
  size_t j = cellStart;
  while (j < cellEnd) {
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
_ComputeMuller03Viscosity(float3 *a, const size_t i, const float3 *pos,
                          const float3 *vel, const float *mass,
                          const float *density, const size_t cellStart,
                          const size_t cellEnd, LaplacianFunc nablaW2) {
  size_t j = cellStart;
  while (j < cellEnd) {
    *a +=
        mass[j] * (vel[j] - vel[i]) / 1000.f * nablaW2(length(pos[i] - pos[j]));
    ++j;
  }
  return;
}

template <typename GradientFunc>
__device__ void
_ComputeArtificialViscosity(float3 *a, const size_t i, const float3 *pos,
                            const float3 *vel, const float *mass,
                            const float *density, const float nu,
                            const float kernelRadius, const size_t cellStart,
                            const size_t cellEnd, GradientFunc nablaW) {
  auto h2 = kernelRadius * kernelRadius;
  size_t j = cellStart;
  while (j < cellEnd) {

    float3 dpij = pos[i] - pos[j];
    float3 dv = vel[i] - vel[j];

    float dot_dvdp = dot(dv, dpij);

    float pij = 10.f * nu / density[j] *
                (dot_dvdp / (lengthSquared(dpij) + 0.01f * h2));
    *a += mass[j] * pij * nablaW(dpij);

    ++j;
  }
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func>
__global__ void _ComputeDensity_CUDA(float *density, const float3 *pos,
                                     const float *mass, const float rho0,
                                     const size_t num, const size_t *cellStart,
                                     const float3 *bPos, const float *bVolume,
                                     const size_t *bCellStart,
                                     const int3 gridSize, Pos2GridXYZ p2xyz,
                                     GridXYZ2GridHash xyz2hash, Func W) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  int3 grid_xyz = p2xyz(pos[i]);

  density[i] = mass[i] * W(0.f);

  __syncthreads();
#pragma unroll
  for (int m = 0; m < 27; __syncthreads(), ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeFluidDensity(&density[i], i, pos, mass, cellStart[hash_idx],
                         cellStart[hash_idx + 1], W);
    _ComputeBoundaryDensity(&density[i], pos[i], bPos, bVolume, rho0,
                            bCellStart[hash_idx], bCellStart[hash_idx + 1], W);
  }

  if (density[i] != density[i])
    printf("sph density nan!! density[i]=%.3f \n", density[i]);

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc, typename LaplacianFunc>
__global__ void _ComputeViscosityTerm_CUDA(
    float3 *acc, const float3 *pos, const float3 *vel, const float *mass,
    const float *density, const float rho0, const float visc, const float bnu,
    const size_t num, const size_t *cellStart, const float3 *bPos,
    const float *bVolume, const size_t *bCellStart, const int3 gridSize,
    Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, GradientFunc nablaW,
    LaplacianFunc nablaW2) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  float3 a = make_float3(0.f);
  int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
  for (int m = 0; m < 27; ++m) {

    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeMuller03Viscosity(&a, i, pos, vel, mass, density,
                              cellStart[hash_idx], cellStart[hash_idx + 1],
                              nablaW2);
  }

  if (a.x != a.x || a.y != a.y || a.z != a.z) {
    printf("_ComputeMuller03Viscosity acc nan!! a=%.3f,%.3f,%.3f \n",
           KIRI_EXPAND_FLOAT3(a));
  }

  acc[i] += visc * a;
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void _ComputeArtificialViscosityTerm_CUDA(
    float3 *acc, const float3 *pos, const float3 *vel, const float *mass,
    const float *density, const float rho0, const float nu, const float bnu,
    const float kernelRadius, const size_t num, const size_t *cellStart,
    const float3 *bPos, const float *bVolume, const size_t *bCellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash,
    GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  float3 a = make_float3(0.f);
  int3 grid_xyz = p2xyz(pos[i]);
  __syncthreads();
#pragma unroll
  for (int m = 0; m < 27; __syncthreads(), ++m) {

    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeArtificialViscosity(&a, i, pos, vel, mass, density, nu,
                                kernelRadius, cellStart[hash_idx],
                                cellStart[hash_idx + 1], nablaW);
    _ComputeBoundaryArtificialViscosity(
        &a, pos[i], bPos, vel[i], density[i], bVolume, bnu, kernelRadius, rho0,
        bCellStart[hash_idx], bCellStart[hash_idx + 1], nablaW);
  }

  if (a.x != a.x || a.y != a.y || a.z != a.z) {
    printf("_ComputeArtificialViscosity acc nan!! a=%.3f,%.3f,%.3f \n",
           KIRI_EXPAND_FLOAT3(a));
  }

  acc[i] += a;
  return;
}

} // namespace KIRI

#endif /* _CUDA_SPH_SOLVER_COMMON_GPU_CUH_ */