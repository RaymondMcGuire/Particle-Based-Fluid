/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-27 10:50:17
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-15 17:54:27
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sph\cuda_dfsph_solver_gpu.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_DFSPH_SOLVER_GPU_CUH_
#define _CUDA_DFSPH_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/sph/cuda_dfsph_solver_common_gpu.cuh>

namespace KIRI
{
  template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
            typename GradientFunc>
  __global__ void
  _ComputeAlpha_CUDA(
      float *alpha,
      const float3 *pos,
      const float *mass,
      const float *density,
      const float rho0,
      const size_t num,
      size_t *cellStart,
      const float3 *bPos,
      const float *bVolume,
      size_t *bCellStart,
      const int3 gridSize,
      Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash,
      GradientFunc nablaW)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    int3 grid_xyz = p2xyz(pos[i]);
    alpha[i] = 0.f;
    float3 grad_pi = make_float3(0.f);
    __syncthreads();

#pragma unroll
    for (size_t m = 0; m < 27; __syncthreads(), ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputeAlpha(&alpha[i], &grad_pi, i, pos, mass, cellStart[hash_idx],
                    cellStart[hash_idx + 1], nablaW);
      _ComputeBoundaryAlpha(&grad_pi, pos[i], bPos, bVolume, rho0,
                            bCellStart[hash_idx], bCellStart[hash_idx + 1],
                            nablaW);
    }

    alpha[i] = -1.f / fmaxf(KIRI_EPSILON, alpha[i] + lengthSquared(grad_pi));

    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
            typename GradientFunc>
  __global__ void _ComputeDivgenceError_CUDA(
      float *stiff,
      float *densityError,
      const float *alpha,
      const float3 *pos,
      const float3 *vel,
      const float *mass,
      const float *density,
      const float rho0,
      const float dt,
      const size_t num,
      size_t *cellStart,
      float3 *bPos,
      float *bVolume,
      size_t *bCellStart,
      const int3 gridSize,
      Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash,
      GradientFunc nablaW)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    int3 grid_xyz = p2xyz(pos[i]);
    auto error = 0.f;
    __syncthreads();
#pragma unroll
    for (size_t m = 0; m < 27; __syncthreads(), ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputeDivergenceError(&error, i, pos, mass, vel, cellStart[hash_idx],
                              cellStart[hash_idx + 1], nablaW);
      _ComputeDivergenceErrorBoundary(&error, pos[i], vel[i], bPos, bVolume,
                                      rho0, bCellStart[hash_idx],
                                      bCellStart[hash_idx + 1], nablaW);
    }

    densityError[i] = fmaxf(error, 0.f);

    if (density[i] + dt * densityError[i] < rho0 && density[i] <= rho0)
      densityError[i] = 0.f;

    stiff[i] = densityError[i] * alpha[i];

    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
            typename GradientFunc>
  __global__ void _CorrectDivergenceByJacobi_CUDA(
      float3 *vel,
      const float *stiff,
      const float3 *pos,
      const float *mass,
      const float rho0,
      const size_t num,
      size_t *cellStart,
      const float3 *bPos,
      const float *bVolume,
      size_t *bCellStart,
      const int3 gridSize,
      Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash,
      GradientFunc nablaW)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    int3 grid_xyz = p2xyz(pos[i]);

    __syncthreads();
#pragma unroll
    for (size_t m = 0; m < 27; __syncthreads(), ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _AdaptVelocitiesByDivergence(&vel[i], i, stiff, pos, mass,
                                   cellStart[hash_idx],
                                   cellStart[hash_idx + 1], nablaW);

      _AdaptVelocitiesBoundaryByDivergence(&vel[i], pos[i], stiff[i], bPos, bVolume, rho0,
                                           bCellStart[hash_idx], bCellStart[hash_idx + 1],
                                           nablaW);
    }

    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
            typename GradientFunc>
  __global__ void _ComputeDensityError_CUDA(
      float *densityError,
      float *stiff,
      const float *alpha,
      const float3 *pos,
      const float3 *vel,
      const float *mass,
      const float *density,
      const float rho0,
      const float dt,
      const size_t num,
      size_t *cellStart,
      float3 *bPos,
      float *bVolume,
      size_t *bCellStart,
      const int3 gridSize,
      Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash,
      GradientFunc nablaW)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    int3 grid_xyz = p2xyz(pos[i]);
    auto error = 0.f;
    __syncthreads();
#pragma unroll
    for (size_t m = 0; m < 27; __syncthreads(), ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputeDivergenceError(&error, i, pos, mass, vel, cellStart[hash_idx],
                              cellStart[hash_idx + 1], nablaW);
      _ComputeDivergenceErrorBoundary(&error, pos[i], vel[i], bPos, bVolume, rho0,
                                      bCellStart[hash_idx], bCellStart[hash_idx + 1],
                                      nablaW);
    }

    densityError[i] = fmaxf(dt * error + density[i] - rho0, 0.f);
    stiff[i] = densityError[i] * alpha[i];

    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
            typename GradientFunc>
  __global__ void _CorrectPressureByJacobi_CUDA(
      float3 *vel,
      const float *stiff,
      const float3 *pos,
      const float *mass,
      const float rho0,
      const float dt,
      const size_t num,
      size_t *cellStart,
      const float3 *bPos,
      const float *bVolume,
      size_t *bCellStart,
      const int3 gridSize,
      Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash,
      GradientFunc nablaW)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    int3 grid_xyz = p2xyz(pos[i]);
    auto a = make_float3(0.0f);
    __syncthreads();
#pragma unroll
    for (size_t m = 0; m < 27; __syncthreads(), ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _AdaptVelocitiesByPressure(&vel[i], i, stiff, pos, mass,
                                 dt, cellStart[hash_idx],
                                 cellStart[hash_idx + 1], nablaW);

      _AdaptVelocitiesBoundaryByPressure(&vel[i], pos[i], stiff[i], bPos, bVolume, rho0, dt,
                                         bCellStart[hash_idx], bCellStart[hash_idx + 1],
                                         nablaW);
    }

    return;
  }

} // namespace KIRI

#endif /* _CUDA_DFSPH_SOLVER_GPU_CUH_ */