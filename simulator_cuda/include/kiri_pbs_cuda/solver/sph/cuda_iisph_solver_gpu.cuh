/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-12-01 23:00:37
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-12-21 17:05:56
 * @FilePath:
 * \Particle-Based-Fluid-Toolkit\simulator_cuda\include\kiri_pbs_cuda\solver\sph\cuda_iisph_solver_gpu.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_IISPH_SOLVER_GPU_CUH_
#define _CUDA_IISPH_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/sph/cuda_iisph_solver_common_gpu.cuh>

namespace KIRI {
template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void _ComputeDiiTerm_CUDA(
    float3 *dii, const float3 *pos, const float *mass, const float *density,
    const float rho0, const size_t num, size_t *cellStart, const float3 *bPos,
    const float *bVolume, size_t *bCellStart, const int3 gridSize,
    Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  int3 grid_xyz = p2xyz(pos[i]);
  dii[i] = make_float3(0.f);

#pragma unroll
  for (size_t m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeDii(&dii[i], i, pos, mass, density[i], cellStart[hash_idx],
                cellStart[hash_idx + 1], nablaW);
    _ComputeBoundaryDii(&dii[i], pos[i], density[i], bPos, bVolume, rho0,
                        bCellStart[hash_idx], bCellStart[hash_idx + 1], nablaW);
  }

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void _ComputeAiiTerm_CUDA(
    float *aii, float *densityAdv, float *pressure, const float3 *dii,
    const float3 *pos, const float3 *vel, const float3 *acc, const float *mass,
    const float *density, const float *lastPressure, const float rho0,
    const float dt, const size_t num, size_t *cellStart, float3 *bPos,
    float *bVolume, size_t *bCellStart, const int3 gridSize, Pos2GridXYZ p2xyz,
    GridXYZ2GridHash xyz2hash, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  int3 grid_xyz = p2xyz(pos[i]);
  aii[i] = 0.f;
  densityAdv[i] = density[i];
  pressure[i] = 0.5f * lastPressure[i];

  float dpi = mass[i] / fmaxf(KIRI_EPSILON, density[i] * density[i]);

#pragma unroll
  for (size_t m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeDivergenceError(&densityAdv[i], i, pos, mass, vel, dt,
                            cellStart[hash_idx], cellStart[hash_idx + 1],
                            nablaW);
    _ComputeDivergenceErrorBoundary(&densityAdv[i], pos[i], vel[i], dt, bPos,
                                    bVolume, rho0, bCellStart[hash_idx],
                                    bCellStart[hash_idx + 1], nablaW);

    _ComputeAii(&aii[i], i, pos, mass, dpi, dii[i], cellStart[hash_idx],
                cellStart[hash_idx + 1], nablaW);
    _ComputeBoundaryAii(&aii[i], pos[i], dpi, dii[i], bPos, bVolume, rho0,
                        bCellStart[hash_idx], bCellStart[hash_idx + 1], nablaW);
  }

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void
_ComputeDijPjTerm_CUDA(float3 *dijpj, const float3 *pos, const float *mass,
                       const float *density, const float *lastPressure,
                       const size_t num, size_t *cellStart, const int3 gridSize,
                       Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash,
                       GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  int3 grid_xyz = p2xyz(pos[i]);
  dijpj[i] = make_float3(0.f);

#pragma unroll
  for (size_t m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeDijPj(&dijpj[i], i, pos, mass, density, lastPressure,
                  cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
  }

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void _CorrectPressureByJacobi_CUDA(
    float *pressure, float *lastPressure, float *densityError, const float *aii,
    const float3 *dijpj, const float3 *dii, const float *densityAdv,
    const float3 *pos, const float *mass, const float *density,
    const float rho0, const float dt, const size_t num, size_t *cellStart,
    const float3 *bPos, const float *bVolume, size_t *bCellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash,
    GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  int3 grid_xyz = p2xyz(pos[i]);

  densityError[i] = 0.f;
  float dpi = mass[i] / fmaxf(KIRI_EPSILON, density[i] * density[i]);

  float sum = 0.f;
#pragma unroll
  for (size_t m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputePressureSumParts(&sum, i, dijpj, dii, lastPressure, pos, mass, dpi,
                             cellStart[hash_idx], cellStart[hash_idx + 1],
                             nablaW);
    _ComputeBoundaryPressureSumParts(&sum, pos[i], dijpj[i], bPos, bVolume,
                                     rho0, bCellStart[hash_idx],
                                     bCellStart[hash_idx + 1], nablaW);
  }

  __syncthreads();

  const float omega = 0.5f;
  const float h2 = dt * dt;
  const float b = rho0 - densityAdv[i];
  const float lpi = lastPressure[i];
  const float denom = aii[i] * h2;
  if (abs(denom) > KIRI_EPSILON)
    pressure[i] =
        max((1.f - omega) * lpi + omega / denom * (b - h2 * sum), 0.f);
  else
    pressure[i] = 0.f;

  if (pressure[i] != 0.f) {
    const float newDensity =
        rho0 * ((aii[i] * pressure[i] + sum) * h2 - b) + rho0;
    densityError[i] = abs(newDensity - rho0);
  }

  lastPressure[i] = pressure[i];
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void _ComputePressureAcceleration_CUDA(
    float3 *pacc, const float3 *pos, const float *mass, const float *density,
    const float *pressure, const float rho0, const size_t num,
    size_t *cellStart, const float3 *bPos, const float *bVolume,
    size_t *bCellStart, const int3 gridSize, Pos2GridXYZ p2xyz,
    GridXYZ2GridHash xyz2hash, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  int3 grid_xyz = p2xyz(pos[i]);
  pacc[i] = make_float3(0.f);
  float dpi = pressure[i] / fmaxf(KIRI_EPSILON, density[i] * density[i]);

#pragma unroll
  for (size_t m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputePressureAcc(&pacc[i], i, pressure, pos, mass, density, dpi,
                        cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
    _ComputeBoundaryPressureAcc(&pacc[i], pos[i], dpi, bPos, bVolume, rho0,
                                bCellStart[hash_idx], bCellStart[hash_idx + 1],
                                nablaW);
  }

  return;
}

} // namespace KIRI

#endif /* _CUDA_IISPH_SOLVER_GPU_CUH_ */