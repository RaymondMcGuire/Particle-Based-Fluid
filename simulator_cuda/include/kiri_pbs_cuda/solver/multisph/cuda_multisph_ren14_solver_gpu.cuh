/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-11 11:44:38
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-15 17:33:11
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\multisph\cuda_multisph_ren14_solver_gpu.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_MULTISPH_REN14_SOLVER_GPU_CUH_
#define _CUDA_MULTISPH_REN14_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/data/cuda_multisph_params.h>
#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver_common_gpu.cuh>

namespace KIRI
{
  __global__ void _ComputeMultiSphRen14MixDensityAndViscosity_CUDA(
      Ren14PhaseDataBlock1 *phaseDataBlock1,
      const Ren14PhaseDataBlock2 *phaseDataBlock2, const float *rho0,
      const float *visc, const size_t num, const size_t phaseNum)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    float rest_mix_density = 0.f;
    float rest_mix_visc = 0.f;
    for (size_t k = 0; k < phaseNum; ++k)
    {
      rest_mix_density += phaseDataBlock2[i].volume_fractions[k] * rho0[k];
      rest_mix_visc += phaseDataBlock2[i].volume_fractions[k] * visc[k];
    }

    __syncthreads();

    phaseDataBlock1[i].rest_mix_density = rest_mix_density;
    phaseDataBlock1[i].rest_mix_viscosity = rest_mix_visc;
    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func>
  __global__ void _ComputeMultiSphRen14AverageDensity_CUDA(
      float *avgDensity, const float3 *pos, const float *mass,
      const Ren14PhaseDataBlock1 *phaseDataBlock1, const size_t num,
      const size_t *cellStart, const float3 *bPos, const float *bVolume,
      const size_t *bCellStart, const int3 gridSize, Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash, Func W)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    int3 grid_xyz = p2xyz(pos[i]);

    avgDensity[i] = mass[i] * W(0.f);

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

      _ComputeFluidDensity(&avgDensity[i], i, pos, mass, cellStart[hash_idx],
                           cellStart[hash_idx + 1], W);
      _ComputeBoundaryDensity(&avgDensity[i], pos[i], bPos, bVolume,
                              phaseDataBlock1[i].rest_mix_density,
                              bCellStart[hash_idx], bCellStart[hash_idx + 1], W);
    }

    return;
  }

  __global__ void
  _ComputeMultiSphRen14Pressure_CUDA(Ren14PhaseDataBlock1 *phaseDataBlock1,
                                     const Ren14PhaseDataBlock2 *phaseDataBlock2,
                                     const float *avgDensity, const size_t num,
                                     const size_t phaseNum, const bool miscible,
                                     const float stiff, const float negativeScale)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    float rest_mix_pressure =
        stiff * (avgDensity[i] - phaseDataBlock1[i].rest_mix_density);

    if (miscible)
      for (size_t k = 0; k < phaseNum; k++)
        phaseDataBlock1[i].phase_pressure[k] =
            phaseDataBlock2[i].volume_fractions[k] * rest_mix_pressure;
    else
      for (size_t k = 0; k < phaseNum; k++)
        phaseDataBlock1[i].phase_pressure[k] = rest_mix_pressure;

    phaseDataBlock1[i].rest_mix_pressure = rest_mix_pressure;

    return;
  }

  template <typename GradientFunc>
  __device__ void ComputePhaseGradientPressure(
      float3 *gp, const size_t i, const size_t k, const float3 *pos,
      const float *mass, const float *avgDensity,
      const Ren14PhaseDataBlock1 *phaseDataBlock1, size_t j, const size_t cellEnd,
      GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      *gp += mass[j] / avgDensity[j] *
             (phaseDataBlock1[i].phase_pressure[k] -
              phaseDataBlock1[j].phase_pressure[k]) *
             nablaW(pos[i] - pos[j]);
      ++j;
    }

    return;
  }

  template <typename GradientFunc>
  __device__ void _ComputePhaseGradientVolumeFraction(
      float3 *gvf, const size_t i, const size_t k, const float3 *pos,
      const float *mass, const float *avgDensity,
      const Ren14PhaseDataBlock2 *phaseDataBlock2, size_t j, const size_t cellEnd,
      GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      *gvf += mass[j] / avgDensity[j] *
              (phaseDataBlock2[i].volume_fractions[k] -
               phaseDataBlock2[j].volume_fractions[k]) *
              nablaW(pos[i] - pos[j]);
      ++j;
    }

    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
            typename GradientFunc>
  __global__ void _ComputeMultiSphRen14GradientTerm_CUDA(
      Ren14PhaseDataBlock1 *phaseDataBlock1,
      Ren14PhaseDataBlock2 *phaseDataBlock2, const float3 *pos, const float *mass,
      const float *avgDensity, const size_t num, const size_t phaseNum,
      const bool miscible, const size_t *cellStart, const int3 gridSize,
      Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, GradientFunc nablaW)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num || !miscible)
      return;

    int3 grid_xyz = p2xyz(pos[i]);

    for (size_t k = 0; k < phaseNum; k++)
    {
      float3 gp = make_float3(0.f);
      float3 gvf = make_float3(0.f);

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

        ComputePhaseGradientPressure(&gp, i, k, pos, mass, avgDensity,
                                     phaseDataBlock1, cellStart[hash_idx],
                                     cellStart[hash_idx + 1], nablaW);
        _ComputePhaseGradientVolumeFraction(&gvf, i, k, pos, mass, avgDensity,
                                            phaseDataBlock2, cellStart[hash_idx],
                                            cellStart[hash_idx + 1], nablaW);
      }

      __syncthreads();

      phaseDataBlock1[i].gradient_pressures[k] = gp;
      phaseDataBlock2[i].gradient_volume_fractions[k] = gvf;
    }

    return;
  }

  __global__ void _ComputeMultiSphRen14DriftVelocities_CUDA(
      Ren14PhaseDataBlock1 *phaseDataBlock1,
      Ren14PhaseDataBlock2 *phaseDataBlock2, const float3 *acc, const float *rho0,
      const size_t num, const size_t phaseNum, const bool miscible,
      const float tou, const float sigma, const float3 gravity)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    float inertia_sum = 0.f;
    float3 pressure_sum = make_float3(0.f);
    float3 diffuse_sum = make_float3(0.f);

    for (size_t k = 0; k < phaseNum; k++)
    {
      float ck = phaseDataBlock2[i].volume_fractions[k] * rho0[k] /
                 phaseDataBlock1[i].rest_mix_density;
      inertia_sum += ck * rho0[k];

      if (miscible)
      {
        pressure_sum += ck * phaseDataBlock1[i].gradient_pressures[k];

        // if (phaseDataBlock2[i].volume_fractions[k] > KIRI_EPSILON)
        //   diffuse_sum += ck * phaseDataBlock2[i].gradient_volume_fractions[k] /
        //                 phaseDataBlock2[i].volume_fractions[k];
      }
    }

    __syncthreads();

    for (size_t k = 0; k < phaseNum; k++)
    {
      float3 drift_velocities = make_float3(0.f);

      float3 interia_parts = tou * (rho0[k] - inertia_sum) * (gravity - acc[i]);
      drift_velocities += interia_parts;

      if (miscible)
      {
        float3 pressure_parts =
            tou * (phaseDataBlock1[i].gradient_pressures[k] - pressure_sum);
        drift_velocities -= pressure_parts;

        // if (phaseDataBlock2[i].volume_fractions[k] > KIRI_EPSILON) {
        //   float3 diffuse_parts =
        //       1e-4f * (phaseDataBlock2[i].gradient_volume_fractions[k] /
        //                    phaseDataBlock2[i].volume_fractions[k] -
        //                diffuse_sum);
        //   drift_velocities -= diffuse_parts;
        // }
      }

      __syncthreads();

      phaseDataBlock2[i].drift_velocities[k] = drift_velocities;
    }

    return;
  }

  __global__ void _UpdateMultiSphRen14VelocityByDriftVelocity_CUDA(
      float3 *vel, const Ren14PhaseDataBlock1 *phaseDataBlock1,
      const Ren14PhaseDataBlock2 *phaseDataBlock2, const size_t num,
      const size_t phaseNum, const float *rho0)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    float3 v = make_float3(0.f);
    for (size_t k = 0; k < phaseNum; k++)
    {
      v += (vel[i] + phaseDataBlock2[i].drift_velocities[k]) *
           phaseDataBlock2[i].volume_fractions[k] * rho0[k];
    }

    __syncthreads();

    vel[i] = v / phaseDataBlock1[i].rest_mix_density;
    return;
  }

  template <typename GradientFunc>
  __device__ void
  _ComputeDVFmotion_of_mixture(float *dvf, const size_t i, const size_t k,
                               const float3 *pos, const float *mass,
                               const float3 *vel, const float *avgDensity,
                               const Ren14PhaseDataBlock2 *phaseDataBlock2, size_t j,
                               const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      *dvf += -mass[j] / avgDensity[j] *
              (phaseDataBlock2[j].volume_fractions[k] +
               phaseDataBlock2[i].volume_fractions[k]) /
              2.f * dot((vel[j] - vel[i]), nablaW(pos[i] - pos[j]));
      ++j;
    }

    return;
  }

  template <typename GradientFunc>
  __device__ void
  _ComputeDVFDiscrepancy(float *dvf, const size_t i, const size_t k,
                         const float3 *pos, const float *mass,
                         const float *avgDensity,
                         const Ren14PhaseDataBlock2 *phaseDataBlock2, size_t j,
                         const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      *dvf += -mass[j] / avgDensity[j] *
              dot((phaseDataBlock2[j].volume_fractions[k] *
                       phaseDataBlock2[j].drift_velocities[k] +
                   phaseDataBlock2[i].volume_fractions[k] *
                       phaseDataBlock2[i].drift_velocities[k]),
                  nablaW(pos[i] - pos[j]));

      ++j;
    }

    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
            typename GradientFunc>
  __global__ void _ComputeMultiSphRen14DeltaVolumeFraction_CUDA(
      Ren14PhaseDataBlock2 *phaseDataBlock2, const float3 *pos, const float *mass,
      const float3 *vel, const float *avgDensity, const size_t num,
      const size_t phaseNum, const bool miscible, const size_t *cellStart,
      const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash,
      GradientFunc nablaW)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num || !miscible)
      return;

    int3 grid_xyz = p2xyz(pos[i]);

    for (size_t k = 0; k < phaseNum; k++)
    {
      float motion_of_mixture = 0.f;
      float discrepancy_between_phase = 0.f;

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

        _ComputeDVFmotion_of_mixture(
            &motion_of_mixture, i, k, pos, mass, vel, avgDensity, phaseDataBlock2,
            cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
        _ComputeDVFDiscrepancy(&discrepancy_between_phase, i, k, pos, mass,
                               avgDensity, phaseDataBlock2, cellStart[hash_idx],
                               cellStart[hash_idx + 1], nablaW);
      }

      phaseDataBlock2[i].delta_volume_fractions[k] = abs(discrepancy_between_phase + motion_of_mixture);
    }

    return;
  }

  __global__ void
  _CorrectMultiSphRen14VolumeFraction_CUDA(
      Ren14PhaseDataBlock1 *phaseDataBlock1,
      Ren14PhaseDataBlock2 *phaseDataBlock2,
      const float *rho0,
      const size_t num,
      const size_t phaseNum,
      const bool miscible,
      const float stiff,
      const float speed,
      const float dt)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num || !miscible)
      return;

    // correct volume fraction
    float sum_volume_fraction = 0.f;
    for (size_t k = 0; k < phaseNum; k++)
    {

      phaseDataBlock2[i].last_volume_fractions[k] = phaseDataBlock2[i].volume_fractions[k];

      phaseDataBlock2[i].volume_fractions[k] += phaseDataBlock2[i].delta_volume_fractions[k] * dt * 30;

      if (phaseDataBlock2[i].volume_fractions[k] > 0.f)
        sum_volume_fraction += phaseDataBlock2[i].volume_fractions[k];
      else
        phaseDataBlock2[i].volume_fractions[k] = 0.f;
    }

    __syncthreads();
    float delta_pressure = 0.f;
    for (size_t k = 0; k < phaseNum; k++)
    {
      if (sum_volume_fraction > 0.f)
      {
        phaseDataBlock2[i].volume_fractions[k] /= sum_volume_fraction;

        float delta_alpha = phaseDataBlock2[i].volume_fractions[k] - phaseDataBlock2[i].last_volume_fractions[k];
        delta_pressure += -stiff * rho0[k] * delta_alpha;
      }
      else
        phaseDataBlock2[i].volume_fractions[k] = phaseDataBlock2[i].last_volume_fractions[k];
    }

    phaseDataBlock1[i].rest_mix_pressure += delta_pressure;

    return;
  }

  template <typename GradientFunc>
  __device__ void _ComputeMultiSphNablaTerm(
      float3 *a, const size_t i, const float3 *pos, const float *mass,
      const Ren14PhaseDataBlock1 *phaseDataBlock1, const float *avgDensity,
      size_t j, const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      *a += -mass[j] *
            (phaseDataBlock1[i].rest_mix_pressure /
                 (avgDensity[i] * avgDensity[i]) +
             phaseDataBlock1[j].rest_mix_pressure /
                 (avgDensity[j] * avgDensity[j])) *
            nablaW(pos[i] - pos[j]);
      ++j;
    }

    return;
  }

  template <typename GradientFunc>
  __device__ void _ComputeMultiSphBoundaryNablaTerm(
      float3 *a, const float3 posi, const float mixDensityi,
      const float mixPressurei, const float3 *bpos, const float *volume, size_t j,
      const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      *a += -volume[j] * (mixPressurei / mixDensityi) * nablaW(posi - bpos[j]);
      ++j;
    }
    return;
  }

  template <typename GradientFunc>
  __device__ void _ComputeMultiSphTDMTerm(
      float3 *a, const size_t i, const size_t k, const float3 *pos,
      const float *mass, const float *avgDensity,
      const Ren14PhaseDataBlock1 *phaseDataBlock1,
      const Ren14PhaseDataBlock2 *phaseDataBlock2, const float rhok, size_t j,
      const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      if (i != j)
      {

        float3 nabla_wij = nablaW(pos[i] - pos[j]);
        float3 dvdfi = phaseDataBlock2[i].volume_fractions[k] *
                       phaseDataBlock2[i].drift_velocities[k] *
                       dot(phaseDataBlock2[i].drift_velocities[k], nabla_wij);
        float3 dvdfj = phaseDataBlock2[j].volume_fractions[k] *
                       phaseDataBlock2[j].drift_velocities[k] *
                       dot(phaseDataBlock2[j].drift_velocities[k], nabla_wij);
        *a += -mass[j] / phaseDataBlock1[i].rest_mix_density / avgDensity[j] *
              (rhok * (dvdfi + dvdfj));
      }
      ++j;
    }
    return;
  }

  template <typename GradientFunc>
  __device__ void
  _ComputeMultiSphTMTerm(float3 *a, const size_t i, const float3 *pos,
                         const float *mass, const float3 *vel,
                         const float *avgDensity,
                         const Ren14PhaseDataBlock1 *phaseDataBlock1,
                         const float kernelRadius, const float soundSpeed,
                         size_t j, const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {

      float3 dpij = pos[i] - pos[j];
      float3 dv = vel[i] - vel[j];

      float dot_dvdp = dot(dv, dpij);
      if (dot_dvdp < 0.f)
      {
        float nu = (phaseDataBlock1[i].rest_mix_viscosity + phaseDataBlock1[j].rest_mix_viscosity) * kernelRadius * soundSpeed / (avgDensity[i] + avgDensity[j]);
        float pij = -nu * (dot_dvdp / (lengthSquared(dpij) + (0.1f * kernelRadius) * (0.1f * kernelRadius)));
        *a += -mass[j] * pij * nablaW(dpij);
      }

      ++j;
    }
    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
            typename GradientFunc>
  __global__ void _ComputeMultiSphRen14Acc_CUDA(
      float3 *acc, const float3 *pos, const float *mass, const float3 *vel,
      const float *avgDensity, const Ren14PhaseDataBlock1 *phaseDataBlock1,
      const Ren14PhaseDataBlock2 *phaseDataBlock2, const float *rho0,
      const float3 gravity, const float kernelRadius, const float soundSpeed,
      const float bnu, const size_t num, const size_t phaseNum,
      const size_t *cellStart, const float3 *bPos, const float *bVolume,
      const size_t *bCellStart, const int3 gridSize, Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash, GradientFunc nablaW)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    float3 a = make_float3(gravity);
    int3 grid_xyz = p2xyz(pos[i]);

    float3 pressure = make_float3(0.f);
    float3 pressure_boundary = make_float3(0.f);
    float3 drift = make_float3(0.f);
    float3 visc = make_float3(0.f);
    float3 visc_boundary = make_float3(0.f);

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

      // pressure
      _ComputeMultiSphNablaTerm(&pressure, i, pos, mass, phaseDataBlock1,
                                avgDensity, cellStart[hash_idx],
                                cellStart[hash_idx + 1], nablaW);
      _ComputeMultiSphBoundaryNablaTerm(
          &pressure_boundary, pos[i], phaseDataBlock1[i].rest_mix_density,
          phaseDataBlock1[i].rest_mix_pressure, bPos, bVolume,
          bCellStart[hash_idx], bCellStart[hash_idx + 1], nablaW);

      // drift
      for (size_t k = 0; k < phaseNum; k++)
        _ComputeMultiSphTDMTerm(
            &drift, i, k, pos, mass, avgDensity, phaseDataBlock1, phaseDataBlock2,
            rho0[k], cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);

      // viscosity
      _ComputeMultiSphTMTerm(&visc, i, pos, mass, vel, avgDensity, phaseDataBlock1,
                             kernelRadius, soundSpeed, cellStart[hash_idx],
                             cellStart[hash_idx + 1], nablaW);

      _ComputeBoundaryArtificialViscosity(
          &visc_boundary, pos[i], bPos, vel[i], avgDensity[i], bVolume, bnu, kernelRadius,
          phaseDataBlock1[i].rest_mix_density, bCellStart[hash_idx],
          bCellStart[hash_idx + 1], nablaW);
    }

    __syncthreads();

    acc[i] = a + pressure + pressure_boundary + visc + visc_boundary + drift;
    return;
  }

  __global__ void _ComputeMultiSphRen14PhaseColorAndMass_CUDA(
      float *mass,
      float3 *color,
      const Ren14PhaseDataBlock1 *phaseDataBlock1,
      const Ren14PhaseDataBlock2 *phaseDataBlock2,
      const size_t num,
      const size_t phaseNum,
      const float *mass0,
      const bool miscible,
      const float3 *color0)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    float m = 0.f;
    float3 col = make_float3(0.f);

    for (size_t k = 0; k < phaseNum; k++)
    {
      m += phaseDataBlock2[i].volume_fractions[k] * mass0[k];
      col += phaseDataBlock2[i].volume_fractions[k] * color0[k];
    }

    __syncthreads();

    if (m != m || m == 0.f)
      printf("m=%.3f; volume_fractions[i]=%.3f,%.3f\n ", m,
             phaseDataBlock2[i].volume_fractions[0],
             phaseDataBlock2[i].volume_fractions[1]);

    mass[i] = m;
    color[i] = col;
    return;
  }

} // namespace KIRI

#endif /* _CUDA_MULTISPH_REN14_SOLVER_GPU_CUH_ */