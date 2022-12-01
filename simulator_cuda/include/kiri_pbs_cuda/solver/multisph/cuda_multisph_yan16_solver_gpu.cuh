/*
 * @Author: Xu.WANG
 * @Date: 2020-07-04 14:48:23
 * @LastEditTime: 2021-04-23 02:20:05
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\multisph\cuda_multisph_yan16_solver_gpu.cuh
 */

#ifndef _CUDA_MULTISPH_YAN16_SOLVER_GPU_CUH_
#define _CUDA_MULTISPH_YAN16_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/data/cuda_multisph_params.h>
#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver_common_gpu.cuh>

namespace KIRI
{

  template <typename GradientFunc>
  __device__ void ComputeMultiSphYanBoundaryNablaTerm(
      float3 *a, const float3 posi, const float mixDensityi,
      const float mixPressurei, const float3 *bpos, const float *volume, size_t j,
      const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      *a += -volume[j] * (mixPressurei / fmaxf(KIRI_EPSILON, mixDensityi)) *
            nablaW(posi - bpos[j]);
      ++j;
    }
    return;
  }

  template <typename GradientFunc>
  __device__ void
  ComputeSolidPhaseVelocityGradient(tensor3x3 *vg, const size_t i, const size_t k,
                                    const float3 *pos, const float3 *vel,
                                    const float *mass, const float *avgDensity,
                                    const Yan16PhaseData *phaseDataBlock1, size_t j,
                                    const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      if (i != j && phaseDataBlock1[j].phase_type != 0 &&
          phaseDataBlock1[j].volume_fractions[k] > 0.5f)
      {
        float volumeParts =
            2.f * (phaseDataBlock1[i].volume_fractions[k] - 0.5f) *
            (phaseDataBlock1[j].volume_fractions[k] - 0.5f) /
            fmaxf(KIRI_EPSILON, phaseDataBlock1[i].volume_fractions[k] - 0.5f +
                                    phaseDataBlock1[j].volume_fractions[k] - 0.5f);
        float3 velocityParts = phaseDataBlock1[j].drift_velocities[k] -
                               phaseDataBlock1[i].drift_velocities[k] + vel[j] - vel[i];
        *vg += mass[j] / fmaxf(KIRI_EPSILON, avgDensity[j]) * volumeParts *
               make_tensor3x3(velocityParts, nablaW(pos[i] - pos[j]));
      }
      ++j;
    }

    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
            typename GradientFunc>
  __global__ void ComputeMultiSphYanSolidDeviatoricStressRateTensor_CUDA(
      Yan16PhaseData *phaseDataBlock1, const float3 *pos, const float *mass,
      const float3 *vel, const float *avgDensity, const float G, const size_t num,
      const size_t phaseNum, const size_t *cellStart, const int3 gridSize,
      Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, GradientFunc nablaW)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    tensor3x3 velocityGradients = make_tensor3x3(make_float3(0.f));
    int3 grid_xyz = p2xyz(pos[i]);
#pragma unroll
    for (size_t m = 0; m < 27; ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      // velocity gradient
      for (size_t k = 0; k < phaseNum; k++)
        ComputeSolidPhaseVelocityGradient(
            &velocityGradients, i, k, pos, vel, mass, avgDensity, phaseDataBlock1,
            cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
    }

    for (size_t k = 0; k < phaseNum; k++)
    {
      tensor3x3 deviatoricStressRateTensor;
      tensor3x3 strainRateTensor = decompose_symmetric(velocityGradients);
      // epslion
      tensor3x3 deviatoricElasticStrainRateTensor =
          deviatoric_tensor(strainRateTensor);
      // omega
      tensor3x3 rotationRateTensor = decompose_antisymmetric(velocityGradients);

      if (phaseDataBlock1[i].phase_type == 1)
      {
        // elastoplastic
        deviatoricStressRateTensor =
            dot(rotationRateTensor, phaseDataBlock1[i].deviatoric_stress_tensor[k]) -
            dot(phaseDataBlock1[i].deviatoric_stress_tensor[k], rotationRateTensor) +
            2.f * G * deviatoricElasticStrainRateTensor;
      }
      else if (phaseDataBlock1[i].phase_type == 2)
      {
        // hypoplastic
        float c1 = 700.f, c2 = 0.f, c3 = 0.5f;
        float p = first_stress_invariants(phaseDataBlock1[i].stress_tensor[k]) / 3.f;

        float absStrainRateTensor =
            sqrt(ddot(strainRateTensor, strainRateTensor));
        deviatoricStressRateTensor =
            -3.f * c1 * p * deviatoricElasticStrainRateTensor -
            2.f * c1 * c3 * phaseDataBlock1[i].deviatoric_stress_tensor[k] *
                absStrainRateTensor +
            c2;
        phaseDataBlock1[i].stress_rate_tensor[k] = 2.f * G * strainRateTensor;
      }

      // record deviatoric stress rate tensor
      phaseDataBlock1[i].deviatoric_stress_rate_tensor[k] = deviatoricStressRateTensor;
    }

    return;
  }

  __global__ void CorrectMultiSphYanSolidDeviatoricStressTensor_CUDA(
      Yan16PhaseData *phaseDataBlock1, const size_t num, const size_t phaseNum,
      const float Y)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num || phaseDataBlock1[i].phase_type == 0)
      return;

    for (size_t k = 0; k < phaseNum; k++)
    {
      if (phaseDataBlock1[i].phase_types[k] == 0 ||
          phaseDataBlock1[i].volume_fractions[k] <= 0.5f)
        continue;

      // update deviatoric Stress Tensor
      phaseDataBlock1[i].deviatoric_stress_tensor[k] +=
          phaseDataBlock1[i].deviatoric_stress_rate_tensor[k] * 0.0005f;
      float deviatoricStressTensorJ2 =
          second_deviatoric_stress_invariants_by_deviatoric_tensor(
              phaseDataBlock1[i].deviatoric_stress_tensor[k]);

      if (phaseDataBlock1[i].phase_type == 1)
      {
        // elastoplastic
        if (sqrt(deviatoricStressTensorJ2) > Y)
          phaseDataBlock1[i].deviatoric_stress_tensor[k] /= Y;
      }
      else if (phaseDataBlock1[i].phase_type == 2)
      {
        phaseDataBlock1[i].stress_tensor[k] +=
            phaseDataBlock1[i].stress_rate_tensor[k] * 0.0005f;
        // hypoplastic
        float p = first_stress_invariants(phaseDataBlock1[i].stress_tensor[k]) / 3.f;
        float tan_phi = tanf(45.f / 180.f * KIRI_PI);
        float alpha_phi = tan_phi / sqrt(9.f + 12.f * tan_phi * tan_phi);
        float c = 1e2f;
        float k_c = 3 * c / sqrt(9.f + 12.f * tan_phi * tan_phi);
        float yield = -alpha_phi * p + k_c;
        if (sqrt(deviatoricStressTensorJ2) > yield &&
            sqrt(deviatoricStressTensorJ2) != 0.f)
          phaseDataBlock1[i].deviatoric_stress_tensor[k] =
              yield * phaseDataBlock1[i].deviatoric_stress_tensor[k] /
              sqrt(deviatoricStressTensorJ2);
      }
    }
    return;
  }

  __global__ void ComputeMultiSphYanMixDensityAndViscosity_CUDA(
      Yan16PhaseData *phaseDataBlock1, const float *rho0, const float *visc,
      const size_t num, const size_t phaseNum)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    float rest_mix_density = 0.f;
    float rest_mix_visc = 0.f;
    for (size_t k = 0; k < phaseNum; ++k)
    {
      rest_mix_density += phaseDataBlock1[i].volume_fractions[k] * rho0[k];
      rest_mix_visc += phaseDataBlock1[i].volume_fractions[k] * visc[k];
    }

    phaseDataBlock1[i].rest_mix_density = rest_mix_density;
    phaseDataBlock1[i].rest_mix_viscosity = rest_mix_visc;
    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func>
  __global__ void ComputeMultiSphYanDensity_CUDA(
      float *avgDensity, const float3 *pos, const float *mass,
      const Yan16PhaseData *phaseDataBlock1, const size_t num, const size_t *cellStart,
      const float3 *bPos, const float *bVolume, const size_t *bCellStart,
      const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, Func W)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
    for (size_t m = 0; m < 27; ++m)
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
                              phaseDataBlock1[i].rest_mix_density, bCellStart[hash_idx],
                              bCellStart[hash_idx + 1], W);
    }

    return;
  }

  __global__ void
  ComputeMultiSphYanPressure_CUDA(Yan16PhaseData *phaseDataBlock1,
                                  const float *avgDensity, const size_t num,
                                  const size_t phaseNum, const bool miscible,
                                  const float stiff, const float negativeScale)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    float rest_mix_pressure =
        stiff * (avgDensity[i] - phaseDataBlock1[i].rest_mix_density);

    if (rest_mix_pressure < 0.f)
      rest_mix_pressure *= negativeScale;

    if (miscible)
      for (size_t k = 0; k < phaseNum; k++)
        phaseDataBlock1[i].phase_pressure[k] =
            phaseDataBlock1[i].volume_fractions[k] * rest_mix_pressure;
    else
      for (size_t k = 0; k < phaseNum; k++)
        phaseDataBlock1[i].phase_pressure[k] = rest_mix_pressure;

    phaseDataBlock1[i].rest_mix_pressure = rest_mix_pressure;

    return;
  }

  template <typename GradientFunc>
  __device__ void ComputeYanPhaseGradientPressure(
      float3 *gp, const size_t i, const size_t k, const float3 *pos,
      const float *mass, const float *avgDensity, const Yan16PhaseData *phaseDataBlock1,
      size_t j, const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      if (i != j)
        *gp += mass[j] / fmaxf(KIRI_EPSILON, avgDensity[j]) *
               (phaseDataBlock1[j].phase_pressure[k] - phaseDataBlock1[i].phase_pressure[k]) *
               nablaW(pos[i] - pos[j]);
      ++j;
    }

    return;
  }

  template <typename GradientFunc>
  __device__ void ComputeYanPhaseGradientVolumeFraction(
      float3 *gvf, const size_t i, const size_t k, const float3 *pos,
      const float *mass, const float *avgDensity, const Yan16PhaseData *phaseDataBlock1,
      size_t j, const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      if (i != j)
        *gvf += mass[j] / fmaxf(KIRI_EPSILON, avgDensity[j]) *
                (phaseDataBlock1[j].volume_fractions[k] -
                 phaseDataBlock1[i].volume_fractions[k]) *
                nablaW(pos[i] - pos[j]);
      ++j;
    }

    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
            typename GradientFunc>
  __global__ void ComputeMultiSphYanGradientTerm_CUDA(
      Yan16PhaseData *phaseDataBlock1, const float3 *pos, const float *mass,
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
#pragma unroll
      for (size_t m = 0; m < 27; ++m)
      {
        int3 cur_grid_xyz =
            grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
        const size_t hash_idx =
            xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
        if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
          continue;

        ComputeYanPhaseGradientPressure(&gp, i, k, pos, mass, avgDensity,
                                        phaseDataBlock1, cellStart[hash_idx],
                                        cellStart[hash_idx + 1], nablaW);
        ComputeYanPhaseGradientVolumeFraction(&gvf, i, k, pos, mass, avgDensity,
                                              phaseDataBlock1, cellStart[hash_idx],
                                              cellStart[hash_idx + 1], nablaW);
      }

      __syncthreads();

      phaseDataBlock1[i].gradient_pressures[k] = gp;
      phaseDataBlock1[i].gradient_volume_fractions[k] = gvf;
    }

    return;
  }

  __global__ void ComputeMultiSphYanDriftVelocities_CUDA(
      Yan16PhaseData *phaseDataBlock1, const float3 *acc, const float *rho0,
      const size_t num, const size_t phaseNum, const bool miscible,
      const float tou, const float sigma, const float3 gravity)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    float inertiaSum = 0.f;
    float3 pressureSum = make_float3(0.f);
    float3 diffuseSum = make_float3(0.f);

    for (size_t k = 0; k < phaseNum; k++)
    {
      float ck = phaseDataBlock1[i].volume_fractions[k] * rho0[k] /
                 phaseDataBlock1[i].rest_mix_density;
      inertiaSum += ck * rho0[k];

      if (miscible)
      {
        pressureSum += ck * phaseDataBlock1[i].gradient_pressures[k];
        // diffuseSum += ck * phaseDataBlock1[i].gradient_volume_fractions[k] /
        // fmaxf(KIRI_EPSILON, phaseDataBlock1[i].volume_fractions[k]);
      }
    }

    __syncthreads();

    for (size_t k = 0; k < phaseNum; k++)
    {
      float3 drift_velocities = tou * (rho0[k] - inertiaSum) * (gravity - acc[i]);

      if (miscible)
      {
        drift_velocities -=
            tou * (phaseDataBlock1[i].gradient_pressures[k] - pressureSum);
        // drift_velocities -= 1e-4f * (phaseDataBlock1[i].gradient_volume_fractions[k]
        // / fmaxf(KIRI_EPSILON, phaseDataBlock1[i].volume_fractions[k]) - diffuseSum);
      }

      __syncthreads();

      phaseDataBlock1[i].drift_velocities[k] = drift_velocities;
    }

    return;
  }

  template <typename GradientFunc>
  __device__ void
  ComputeYanDVFMotionOfMixture(float *dvf, const size_t i, const size_t k,
                               const float3 *pos, const float *mass,
                               const float3 *vel, const float *avgDensity,
                               const Yan16PhaseData *phaseDataBlock1, size_t j,
                               const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      if (i != j)
        *dvf += -mass[j] / fmaxf(KIRI_EPSILON, avgDensity[j]) *
                (phaseDataBlock1[j].volume_fractions[k] +
                 phaseDataBlock1[i].volume_fractions[k]) /
                2.f * dot((vel[i] - vel[j]), nablaW(pos[i] - pos[j]));
      ++j;
    }

    return;
  }

  template <typename GradientFunc>
  __device__ void ComputeYanDVFDiscrepancy(
      float *dvf, const size_t i, const size_t k, const float3 *pos,
      const float *mass, const float *avgDensity, const Yan16PhaseData *phaseDataBlock1,
      size_t j, const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      if (i != j)
        *dvf += -mass[j] / fmaxf(KIRI_EPSILON, avgDensity[j]) *
                dot((phaseDataBlock1[j].volume_fractions[k] *
                         phaseDataBlock1[j].drift_velocities[k] +
                     phaseDataBlock1[i].volume_fractions[k] *
                         phaseDataBlock1[i].drift_velocities[k]),
                    nablaW(pos[i] - pos[j]));
      ++j;
    }

    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
            typename GradientFunc>
  __global__ void ComputeMultiSphYanDeltaVolumeFraction_CUDA(
      Yan16PhaseData *phaseDataBlock1, const float3 *pos, const float *mass,
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
      float dvf = 0.f;
#pragma unroll
      for (size_t m = 0; m < 27; ++m)
      {
        int3 cur_grid_xyz =
            grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
        const size_t hash_idx =
            xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
        if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
          continue;

        ComputeYanDVFMotionOfMixture(&dvf, i, k, pos, mass, vel, avgDensity,
                                     phaseDataBlock1, cellStart[hash_idx],
                                     cellStart[hash_idx + 1], nablaW);
        ComputeYanDVFDiscrepancy(&dvf, i, k, pos, mass, avgDensity, phaseDataBlock1,
                                 cellStart[hash_idx], cellStart[hash_idx + 1],
                                 nablaW);
      }

      __syncthreads();

      // abs(-motionOfMixture - discrepancyBetweenPhase);
      phaseDataBlock1[i].delta_volume_fractions[k] = dvf;
    }

    return;
  }

  __global__ void CorrectMultiSphYanVolumeFraction_CUDA(Yan16PhaseData *phaseDataBlock1,
                                                        const size_t num,
                                                        const size_t phaseNum,
                                                        const bool miscible,
                                                        const float dt)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num || !miscible)
      return;

    // correct volume fraction
    float sumVolumeFraction = 0.f;
    for (size_t k = 0; k < phaseNum; k++)
    {
      float vf = phaseDataBlock1[i].volume_fractions[k] +
                 phaseDataBlock1[i].delta_volume_fractions[k] * dt * 5.f;

      if (vf > 0.f)
        sumVolumeFraction += vf;
      else
        vf = 0.f;

      __syncthreads();
      phaseDataBlock1[i].volume_fractions[k] = vf;
    }

    __syncthreads();

    // fluid -> fluid
    if (phaseDataBlock1[i].phase_type == 0 && phaseDataBlock1[i].last_phase_type == 0)
    {
      for (size_t k = 0; k < phaseNum; k++)
      {
        if (sumVolumeFraction > 0.f)
        {
          // rescale volume fraction
          phaseDataBlock1[i].volume_fractions[k] /=
              fmaxf(KIRI_EPSILON, sumVolumeFraction);

          // float deltaAlpha = phaseDataBlock1[i].volume_fractions[k] -
          // phaseDataBlock1[i].last_volume_fractions[k]; deltaPressure += -kappa *
          // restDensities[k] * deltaAlpha;
        }
        else
          phaseDataBlock1[i].volume_fractions[k] =
              phaseDataBlock1[i].last_volume_fractions[k];
      }
    }
    else if (phaseDataBlock1[i].phase_type != 0 &&
             phaseDataBlock1[i].last_phase_type != 0)
    {
      // solid -> solid
      for (size_t k = 0; k < phaseNum; k++)
      {
        if (sumVolumeFraction > 0.f)
        {
          // rescale volume fraction
          phaseDataBlock1[i].volume_fractions[k] /= sumVolumeFraction;

          // fluid -> solid
          if (phaseDataBlock1[i].phase_types[k] != 0 &&
              phaseDataBlock1[i].volume_fractions[k] <= 0.5f)
            phaseDataBlock1[i].phase_type = 0;

          // float deltaAlpha = phaseDataBlock1[i].volume_fractions[k] -
          // phaseDataBlock1[i].last_volume_fractions[k];
          // deltaPressure += -kappa * restDensities[k] * deltaAlpha;
        }
        else
          phaseDataBlock1[i].volume_fractions[k] =
              phaseDataBlock1[i].last_volume_fractions[k];
      }
    }
    else if (phaseDataBlock1[i].phase_type == 0 &&
             phaseDataBlock1[i].last_phase_type != 0)
    {
      // solid -> fluid

      bool recal = false;
      size_t solidIdx = 999;
      for (size_t k = 0; k < phaseNum; k++)
      {
        if (sumVolumeFraction > 0.f)
        {
          if (phaseDataBlock1[i].phase_types[k] != 0 &&
              phaseDataBlock1[i].volume_fractions[k] / sumVolumeFraction > 0.5f)
          {
            recal = true;
            solidIdx = k;
          }
        }
        else
          phaseDataBlock1[i].volume_fractions[k] =
              phaseDataBlock1[i].last_volume_fractions[k];
      }

      if (recal)
      {
        sumVolumeFraction -= phaseDataBlock1[i].volume_fractions[solidIdx];
        phaseDataBlock1[i].volume_fractions[solidIdx] = 0.5f;

        for (size_t k = 0; k < phaseNum; k++)
        {
          if (k != solidIdx)
          {
            // rescale volume fraction
            phaseDataBlock1[i].volume_fractions[k] =
                phaseDataBlock1[i].volume_fractions[k] / sumVolumeFraction * 0.5f;
          }

          // float deltaAlpha = phaseDataBlock1[i].volume_fractions[k] -
          // phaseDataBlock1[i].last_volume_fractions[k]; deltaPressure += -kappa *
          // restDensities[k] * deltaAlpha;
        }
      }
      else
      {
        for (size_t k = 0; k < phaseNum; k++)
        {
          if (sumVolumeFraction > 0.f)
          {
            // rescale volume fraction
            phaseDataBlock1[i].volume_fractions[k] /= sumVolumeFraction;
            // float deltaAlpha = phaseDataBlock1[i].volume_fractions[k] -
            // phaseDataBlock1[i].last_volume_fractions[k];
            // deltaPressure += -kappa * restDensities[k] * deltaAlpha;
          }
          else
            phaseDataBlock1[i].volume_fractions[k] =
                phaseDataBlock1[i].last_volume_fractions[k];
        }
      }
    }

    return;
  }

  template <typename GradientFunc>
  __device__ void
  ComputeMultiSphYanNablaTerm(float3 *a, const size_t i, const float3 *pos,
                              const float *mass, const Yan16PhaseData *phaseDataBlock1,
                              const float *avgDensity, size_t j,
                              const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      if (i != j)
        *a += -mass[j] *
              (phaseDataBlock1[i].rest_mix_pressure /
                   fmaxf(KIRI_EPSILON, avgDensity[i] * avgDensity[i]) +
               phaseDataBlock1[j].rest_mix_pressure /
                   fmaxf(KIRI_EPSILON, avgDensity[j] * avgDensity[j])) *
              nablaW(pos[i] - pos[j]);
      ++j;
    }

    return;
  }

  template <typename GradientFunc>
  __device__ void ComputeMultiSphYanTDMTerm(
      float3 *a, const size_t i, const size_t k, const float3 *pos,
      const float *mass, const float *avgDensity, const Yan16PhaseData *phaseDataBlock1,
      const float rhok, size_t j, const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      if (i != j)
      {
        float3 nabla_wij = nablaW(pos[i] - pos[j]);
        float3 dvdfi = phaseDataBlock1[i].volume_fractions[k] *
                       phaseDataBlock1[i].drift_velocities[k] *
                       dot(phaseDataBlock1[i].drift_velocities[k], nabla_wij);
        float3 dvdfj = phaseDataBlock1[j].volume_fractions[k] *
                       phaseDataBlock1[j].drift_velocities[k] *
                       dot(phaseDataBlock1[j].drift_velocities[k], nabla_wij);
        *a += -mass[j] / phaseDataBlock1[i].rest_mix_density /
              fmaxf(KIRI_EPSILON, avgDensity[j]) * (rhok * (dvdfi + dvdfj));
      }
      ++j;
    }
    return;
  }

  template <typename GradientFunc>
  __device__ void ComputeMultiSphYanTMTerm(
      float3 *a, const size_t i, const float3 *pos, const float *mass,
      const float3 *vel, const float *avgDensity, const Yan16PhaseData *phaseDataBlock1,
      const float kernelRadius, const float soundSpeed, size_t j,
      const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      if (i != j)
      {
        float3 nabla_wij = nablaW(pos[i] - pos[j]);
        float3 dpij = pos[i] - pos[j];
        float3 dvij = vel[i] - vel[j];
        float dotDvDp = dot(dvij, dpij);
        if (dotDvDp < 0.f)
        {
          float nu = (phaseDataBlock1[i].rest_mix_viscosity +
                      phaseDataBlock1[j].rest_mix_viscosity) *
                     kernelRadius * soundSpeed;
          float pij = -nu / fmaxf(KIRI_EPSILON, avgDensity[i] + avgDensity[j]) *
                      (dotDvDp / fmaxf(KIRI_EPSILON, lengthSquared(dpij)));
          *a += -mass[j] * pij * nabla_wij;
        }
      }
      ++j;
    }
    return;
  }

  template <typename GradientFunc>
  __device__ void
  ComputeMultiSphYanSMTerm(float3 *a, const size_t i, const size_t k,
                           const float3 *pos, const float *avgDensity,
                           const Yan16PhaseData *phaseDataBlock1, size_t j,
                           const size_t cellEnd, GradientFunc nablaW)
  {
    while (j < cellEnd)
    {
      if (i != j)
      {
        if (phaseDataBlock1[i].phase_type != 0 && phaseDataBlock1[i].phase_types[k] != 0 &&
            phaseDataBlock1[i].volume_fractions[k] > 0.5f &&
            phaseDataBlock1[j].phase_type != 0 && phaseDataBlock1[j].phase_types[k] != 0 &&
            phaseDataBlock1[j].volume_fractions[k] > 0.5f)
        {
          float volumeParts =
              2.f * (phaseDataBlock1[i].volume_fractions[k] - 0.5f) *
              (phaseDataBlock1[j].volume_fractions[k] - 0.5f) /
              fmaxf(KIRI_EPSILON, phaseDataBlock1[i].volume_fractions[k] - 0.5f +
                                      phaseDataBlock1[j].volume_fractions[k] - 0.5f);
          tensor3x3 deviatoricStressTensorParts =
              phaseDataBlock1[i].deviatoric_stress_tensor[k] /
                  fmaxf(KIRI_EPSILON, avgDensity[i] * avgDensity[i]) +
              phaseDataBlock1[j].deviatoric_stress_tensor[k] /
                  fmaxf(KIRI_EPSILON, avgDensity[j] * avgDensity[j]);
          *a += volumeParts *
                dot(deviatoricStressTensorParts, nablaW(pos[i] - pos[j]));
        }
      }
      ++j;
    }
    return;
  }

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
            typename GradientFunc>
  __global__ void ComputeMultiSphYanAcc_CUDA(
      float3 *acc, const float3 *pos, const float *mass, const float3 *vel,
      const float *avgDensity, const Yan16PhaseData *phaseDataBlock1, const float *rho0,
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
#pragma unroll
    for (size_t m = 0; m < 27; ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      // pressure
      ComputeMultiSphYanNablaTerm(&a, i, pos, mass, phaseDataBlock1, avgDensity,
                                  cellStart[hash_idx], cellStart[hash_idx + 1],
                                  nablaW);
      ComputeMultiSphYanBoundaryNablaTerm(
          &a, pos[i], phaseDataBlock1[i].rest_mix_density,
          phaseDataBlock1[i].rest_mix_pressure, bPos, bVolume, bCellStart[hash_idx],
          bCellStart[hash_idx + 1], nablaW);

      // drift
      for (size_t k = 0; k < phaseNum; k++)
      {
        ComputeMultiSphYanTDMTerm(&a, i, k, pos, mass, avgDensity, phaseDataBlock1,
                                  rho0[k], cellStart[hash_idx],
                                  cellStart[hash_idx + 1], nablaW);
        ComputeMultiSphYanSMTerm(&a, i, k, pos, avgDensity, phaseDataBlock1,
                                 cellStart[hash_idx], cellStart[hash_idx + 1],
                                 nablaW);
      }

      // viscosity
      ComputeMultiSphYanTMTerm(&a, i, pos, mass, vel, avgDensity, phaseDataBlock1,
                               kernelRadius, soundSpeed, cellStart[hash_idx],
                               cellStart[hash_idx + 1], nablaW);
      //_ComputeBoundaryArtificialViscosity(&a, pos[i], bPos, vel[i], avgDensity[i], bVolume,
      //                          bnu, phaseDataBlock1[i].rest_mix_density,
      //                          bCellStart[hash_idx], bCellStart[hash_idx + 1],
      //                          nablaW);
    }
    acc[i] = a;
    return;
  }

  __global__ void ComputeMultiSphYanNextTimeStepData_CUDA(
      float *mass, float3 *vel, float3 *color, Yan16PhaseData *phaseDataBlock1,
      const size_t num, const size_t phaseNum, const float *rho0,
      const float *mass0, const float3 *color0)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    float m = 0.f;
    float3 col = make_float3(0.f);
    float3 v = make_float3(0.f);
    for (size_t k = 0; k < phaseNum; k++)
    {
      // float ck = phaseDataBlock1[i].volume_fractions[k] * rho0[k] /
      // phaseDataBlock1[i].rest_mix_density;
      m += phaseDataBlock1[i].volume_fractions[k] * mass0[k];
      col += phaseDataBlock1[i].volume_fractions[k] * color0[k];
      v += (vel[i] + phaseDataBlock1[i].drift_velocities[k]) *
           phaseDataBlock1[i].volume_fractions[k] * rho0[k];
    }

    __syncthreads();

    mass[i] = m;
    color[i] = col;
    vel[i] = v / phaseDataBlock1[i].rest_mix_density;
    return;
  }

} // namespace KIRI

#endif /* _CUDA_MULTISPH_YAN16_SOLVER_GPU_CUH_ */