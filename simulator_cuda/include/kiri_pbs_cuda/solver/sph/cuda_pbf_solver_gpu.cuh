/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-12 14:47:30
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-20 10:50:01
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sph\cuda_pbf_solver_gpu.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_PBF_SOLVER_GPU_CUH_
#define _CUDA_PBF_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver_common_gpu.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_pbf_solver_common_gpu.cuh>

namespace KIRI
{

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func>
  __global__ void _ComputePBFDensity_CUDA(
      float *density,
      float *densityError,
      const float3 *pos,
      const float *mass,
      const float rho0,
      const size_t num,
      const size_t *cellStart,
      const float3 *posB,
      const float *volumeB,
      const size_t *cellStartB,
      const int3 gridSize,
      Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash,
      Func W)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    int3 grid_xyz = p2xyz(pos[i]);
    density[i] = mass[i] * W(0.f);

    __syncthreads();
#pragma unroll
    for (int m = 0; m < 27; __syncthreads(), ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputeFluidDensity(&density[i], i, pos, mass, cellStart[hash_idx],
                           cellStart[hash_idx + 1], W);
      _ComputeBoundaryDensity(&density[i], pos[i], posB, volumeB, rho0,
                              cellStartB[hash_idx], cellStartB[hash_idx + 1], W);
    }

    if (density[i] != density[i])
      printf("pbf density nan!! density[i]=%.3f \n", density[i]);

    atomicAdd(&densityError[0], max(density[i], rho0) - rho0);

    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
            typename GradientFunc>
  __global__ void
  _ComputePBFLambdaIncompressible_CUDA(
      float *lambda,
      const float *density,
      const float3 *pos,
      const float *mass,
      const float rho0,
      const size_t num,
      const size_t *cellStart,
      const float3 *posB,
      const float *volumeB,
      const size_t *cellStartB,
      const int3 gridSize,
      Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash,
      GradientFunc nablaW)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;
    const float constraint = std::max(density[i] / rho0 - 1.f, 0.f);
    if (constraint == 0.f)
    {
      lambda[i] = 0.f;
      return;
    }

    int3 grid_xyz = p2xyz(pos[i]);

    float3 grad_ci = make_float3(0.f);
    float sum_grad_c2 = 0.f;
    __syncthreads();
#pragma unroll
    for (int m = 0; m < 27; __syncthreads(), ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputePBFGradC(&sum_grad_c2, &grad_ci, i, rho0, pos, mass, cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
      _ComputePBFBoundaryGradC(&grad_ci, pos[i], 1.f, posB, volumeB, cellStartB[hash_idx], cellStartB[hash_idx + 1], nablaW);
    }

    sum_grad_c2 += lengthSquared(grad_ci);
    lambda[i] = -constraint / (sum_grad_c2 + KIRI_EPSILON);

    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
            typename GradientFunc>
  __global__ void
  _ComputePBFLambdaRealtime_CUDA(
      float *lambda,
      const float *density,
      const float3 *pos,
      const float *mass,
      const float rho0,
      const float lambdaEPS,
      const size_t num,
      const size_t *cellStart,
      const float3 *posB,
      const float *volumeB,
      const size_t *cellStartB,
      const int3 gridSize,
      Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash,
      GradientFunc nablaW)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    const float constraint = density[i] / rho0 - 1.f;

    int3 grid_xyz = p2xyz(pos[i]);

    float3 grad_ci = make_float3(0.f);
    float sum_grad_c2 = 0.f;
    __syncthreads();
#pragma unroll
    for (int m = 0; m < 27; __syncthreads(), ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputePBFGradC(&sum_grad_c2, &grad_ci, i, rho0, pos, mass, cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
      _ComputePBFBoundaryGradC(&grad_ci, pos[i], rho0, posB, volumeB, cellStartB[hash_idx], cellStartB[hash_idx + 1], nablaW);
    }

    sum_grad_c2 += lengthSquared(grad_ci);
    lambda[i] = -constraint / (sum_grad_c2 + lambdaEPS);

    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
            typename Func, typename GradientFunc>
  __global__ void _SolvePBFDensityConstrainRealtime_CUDA(
      float3 *deltaPos,
      const float *lambda,
      const float *density,
      const float3 *pos,
      const float *mass,
      const float rho0,
      const float deltaQ,
      const float corrK,
      const float corrN,
      const float dt,
      const size_t num,
      const size_t *cellStart,
      const float3 *posB,
      const float *volumeB,
      const size_t *cellStartB,
      const int3 gridSize,
      Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash,
      Func W,
      GradientFunc nablaW)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    deltaPos[i] = make_float3(0.f);
    int3 grid_xyz = p2xyz(pos[i]);

    __syncthreads();
#pragma unroll
    for (int m = 0; m < 27; __syncthreads(), ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputePBFDeltaPosRealtime(&deltaPos[i], i, rho0, lambda, pos, mass, deltaQ, corrK, corrN, dt, cellStart[hash_idx], cellStart[hash_idx + 1], W, nablaW);
    }

    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename GradientFunc>
  __global__ void _SolvePBFDensityConstrainIncompressible_CUDA(
      float3 *deltaPos,
      const float *lambda,
      const float *density,
      const float3 *pos,
      const float *mass,
      const float rho0,
      const size_t num,
      const size_t *cellStart,
      const float3 *posB,
      const float *volumeB,
      const size_t *cellStartB,
      const int3 gridSize,
      Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash,
      GradientFunc nablaW)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    deltaPos[i] = make_float3(0.f);
    int3 grid_xyz = p2xyz(pos[i]);

    __syncthreads();
#pragma unroll
    for (int m = 0; m < 27; __syncthreads(), ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputePBFDeltaPosIncompressible(&deltaPos[i], i, rho0, lambda, pos, mass, cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);

      _ComputePBFBoundaryDeltaPosIncompressible(&deltaPos[i], pos[i], lambda[i], posB, volumeB, cellStartB[hash_idx], cellStartB[hash_idx + 1], nablaW);
    }

    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func>
  __global__ void _ComputeViscosityXSPH_CUDA(
      float3 *acc, const float3 *pos, const float3 *vel, const float *density,
      const float *mass, const float rho0, const float visc,
      const float boundaryVisc, const float dt, const size_t num,
      const size_t *cellStart, const float3 *posB, const float *volumeB,
      const size_t *cellStartB, const int3 gridSize, Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash, Func W)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    float3 xsph_vel = make_float3(0.f);

    int3 grid_xyz = p2xyz(pos[i]);
    __syncthreads();
#pragma unroll
    for (int m = 0; m < 27; __syncthreads(), ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputeViscosityXSPH(&xsph_vel, i, pos, vel, mass, density, visc,
                            cellStart[hash_idx], cellStart[hash_idx + 1], W);

      _ComputeBoundaryViscosityXSPH(
          &xsph_vel, pos[i], vel[i], density[i], posB, volumeB, rho0,
          boundaryVisc, cellStartB[hash_idx], cellStartB[hash_idx + 1], W);
    }

    acc[i] += 1.f / dt * xsph_vel;
    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename GradientFunc>
  __global__ void _ComputeVorticityOmega_CUDA(
      float3 *omega,
      float3 *normOmega,
      const float3 *pos,
      const float3 *vel,
      const float *density,
      const float *mass,
      const size_t num,
      const size_t *cellStart,
      const int3 gridSize,
      Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash,
      GradientFunc nablaW)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    omega[i] = make_float3(0.f);
    normOmega[i] = make_float3(0.f);

    int3 grid_xyz = p2xyz(pos[i]);
    __syncthreads();
#pragma unroll
    for (int m = 0; m < 27; __syncthreads(), ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputeVorticityOmega(&omega[i], i, pos, vel, mass, density,
                             cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
    }

    if (length(omega[i]) != 0.f)
      normOmega[i] = normalize(omega[i]);

    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename GradientFunc>
  __global__ void _ComputeVorticityConfinement_CUDA(
      float3 *acc,
      const float3 *omega,
      const float3 *normOmega,
      const float3 *pos,
      const float *density,
      const float *mass,
      const float coeff,
      const size_t num,
      const size_t *cellStart,
      const int3 gridSize,
      Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash,
      GradientFunc nablaW)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    float3 eta = make_float3(0.f);

    int3 grid_xyz = p2xyz(pos[i]);
    __syncthreads();
#pragma unroll
    for (int m = 0; m < 27; __syncthreads(), ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputeVorticityConfinement(&eta, i, normOmega, pos, mass, density,
                                   cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
    }

    if (length(eta) != 0.f)
      eta = normalize(eta);

    acc[i] += coeff * cross(eta, omega[i]);
    return;
  }

} // namespace KIRI

#endif /* _CUDA_PBF_SOLVER_GPU_CUH_ */