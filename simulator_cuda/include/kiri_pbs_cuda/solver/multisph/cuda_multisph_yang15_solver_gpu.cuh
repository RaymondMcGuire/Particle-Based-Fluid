/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-24 14:09:08
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-24 14:43:09
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\multisph\cuda_multisph_yang15_solver_gpu.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_MULTISPH_YANG15_SOLVER_GPU_CUH_
#define _CUDA_MULTISPH_YANG15_SOLVER_GPU_CUH_

#include <kiri_pbs_cuda/data/cuda_multisph_params.h>
#include <kiri_pbs_cuda/solver/sph/cuda_pbf_solver_common_gpu.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver_common_gpu.cuh>

namespace KIRI {
__global__ void _ComputeMultiSphYang15AggregateData_CUDA(
    float *mass, Yang15PhaseDataBlock1 *phaseDataBlock1, const float *rho0,
    const float *mass0, const size_t num, const size_t phaseNum) {
  size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  float aggregate_density = 0.f;
  float aggregate_mass = 0.f;
  for (size_t k = 0; k < phaseNum; k++) {
    aggregate_density += phaseDataBlock1[i].mass_ratios[k] / rho0[k];
    aggregate_mass += phaseDataBlock1[i].mass_ratios[k] * mass0[k];
  }
  __syncthreads();

  phaseDataBlock1[i].aggregate_density = 1.f / aggregate_density;
  mass[i] = aggregate_mass;

  if (phaseDataBlock1[i].aggregate_density !=
      phaseDataBlock1[i].aggregate_density)
    printf("phaseDataBlock1[i].aggregate_density nan!! "
           "phaseDataBlock1[i].aggregate_density=%.3f\n",
           phaseDataBlock1[i].aggregate_density);

  if (mass[i] != mass[i])
    printf("mass[i] nan!! mass[i]=%.3f\n", mass[i]);
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func>
__global__ void _ComputeMultiSphYang15AggregateDensity_CUDA(
    float *avgDensity, float *densityError, const float3 *pos,
    const float *mass, const Yang15PhaseDataBlock1 *phaseDataBlock1,
    const size_t num, const size_t *cellStart, const float3 *bPos,
    const float *bVolume, const size_t *bCellStart, const int3 gridSize,
    Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, Func W) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  int3 grid_xyz = p2xyz(pos[i]);
  avgDensity[i] = mass[i] * W(0.f);

  __syncthreads();

#pragma unroll
  for (size_t m = 0; m < 27; __syncthreads(), ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeFluidDensity(&avgDensity[i], i, pos, mass, cellStart[hash_idx],
                         cellStart[hash_idx + 1], W);
    _ComputeBoundaryDensity(&avgDensity[i], pos[i], bPos, bVolume,
                            phaseDataBlock1[i].aggregate_density,
                            bCellStart[hash_idx], bCellStart[hash_idx + 1], W);
  }

  return;
}

template <typename GradientFunc>
__device__ void
_ComputeMassRatioGradient(float3 *massRatioGradient, const size_t i,
                          const size_t k, const float3 *pos, const float *mass,
                          const float *avgDensity,
                          const Yang15PhaseDataBlock1 *phaseDataBlock1,
                          size_t j, const size_t cellEnd, GradientFunc nablaW) {
  while (j < cellEnd) {
    *massRatioGradient +=
        mass[j] * avgDensity[i] *
        (phaseDataBlock1[i].mass_ratios[k] / (avgDensity[i] * avgDensity[i]) +
         phaseDataBlock1[j].mass_ratios[k] / (avgDensity[j] * avgDensity[j])) *
        nablaW(pos[i] - pos[j]);
    ++j;
  }
  return;
}

template <typename GradientFunc>
__device__ void _ComputeMassRatioLaplacian(
    float *massRatioLaplacian, const size_t i, const size_t k,
    const float3 *pos, const float *mass, const float *avgDensity,
    const float eta, const Yang15PhaseDataBlock1 *phaseDataBlock1, size_t j,
    const size_t cellEnd, GradientFunc nablaW) {
  while (j < cellEnd) {

    float3 dp = pos[i] - pos[j];
    *massRatioLaplacian +=
        2.f * mass[j] / avgDensity[j] *
        (phaseDataBlock1[i].mass_ratios[k] -
         phaseDataBlock1[j].mass_ratios[k]) *
        (dot(dp, nablaW(dp)) / (lengthSquared(dp) + eta * eta));

    ++j;
  }
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void _ComputeMultiSphYang15MassRatioTerm_CUDA(
    Yang15PhaseDataBlock2 *phaseDataBlock2,
    const Yang15PhaseDataBlock1 *phaseDataBlock1, const float3 *pos,
    const float *mass, const float *avgDensity, const float eta,
    const size_t num, const size_t phaseNum, const size_t *cellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash,
    GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  int3 grid_xyz = p2xyz(pos[i]);

  for (size_t k = 0; k < phaseNum; k++) {
    float3 mass_ratio_gradient = make_float3(0.f);
    float mass_ratio_laplacian = 0.f;

#pragma unroll
    for (size_t m = 0; m < 27; ++m) {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputeMassRatioGradient(
          &mass_ratio_gradient, i, k, pos, mass, avgDensity, phaseDataBlock1,
          cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);

      _ComputeMassRatioLaplacian(&mass_ratio_laplacian, i, k, pos, mass,
                                 avgDensity, eta, phaseDataBlock1,
                                 cellStart[hash_idx], cellStart[hash_idx + 1],
                                 nablaW);
    }

    phaseDataBlock2[i].mass_ratio_gradient[k] = mass_ratio_gradient;
    phaseDataBlock2[i].mass_ratio_laplacian[k] = mass_ratio_laplacian;

    if (phaseDataBlock2[i].mass_ratio_gradient[k].x !=
            phaseDataBlock2[i].mass_ratio_gradient[k].x ||
        phaseDataBlock2[i].mass_ratio_gradient[k].y !=
            phaseDataBlock2[i].mass_ratio_gradient[k].y ||
        phaseDataBlock2[i].mass_ratio_gradient[k].z !=
            phaseDataBlock2[i].mass_ratio_gradient[k].z) {
      printf("phaseDataBlock2[i].mass_ratio_gradient[k] acc nan!! "
             "phaseDataBlock2[i].mass_ratio_gradient[k]=%.3f,%.3f,%.3f \n",
             KIRI_EXPAND_FLOAT3(phaseDataBlock2[i].mass_ratio_gradient[k]));
    }

    if (phaseDataBlock2[i].mass_ratio_laplacian[k] !=
        phaseDataBlock2[i].mass_ratio_laplacian[k])
      printf("phaseDataBlock2[i].mass_ratio_laplacian[k] nan!! "
             "phaseDataBlock2[i].mass_ratio_laplacian[k]=%.3f\n",
             phaseDataBlock2[i].mass_ratio_laplacian[k]);
  }

  return;
}

__global__ void _ComputeMultiSphYang15ChemicalPotential_CUDA(
    Yang15PhaseDataBlock2 *phaseDataBlock2,
    const Yang15PhaseDataBlock1 *phaseDataBlock1, const size_t num,
    const float alpha, const float s1, const float s2, const float epsilon) {

  size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if (i >= num)
    return;

  float f_term0 = 2 * alpha * (phaseDataBlock1[i].mass_ratios[0] - s1) *
                  (phaseDataBlock1[i].mass_ratios[1] - s2) *
                  (phaseDataBlock1[i].mass_ratios[1] - s2);
  float f_term1 = 2 * alpha * (phaseDataBlock1[i].mass_ratios[1] - s2) *
                  (phaseDataBlock1[i].mass_ratios[0] - s1) *
                  (phaseDataBlock1[i].mass_ratios[0] - s1);

  float s_term0 =
      epsilon * epsilon * phaseDataBlock2[i].mass_ratio_laplacian[0];
  float s_term1 =
      epsilon * epsilon * phaseDataBlock2[i].mass_ratio_laplacian[1];

  float t_term = -0.5f * (f_term0 + f_term1);

  phaseDataBlock2[i].chemical_potential[0] = f_term0 - s_term0 + t_term;
  phaseDataBlock2[i].chemical_potential[1] = f_term1 - s_term1 + t_term;

  if (phaseDataBlock2[i].chemical_potential[0] !=
      phaseDataBlock2[i].chemical_potential[0])
    printf("phaseDataBlock2[i].chemical_potential[0]nan!! "
           "phaseDataBlock2[i].chemical_potential[0]=%.3f\n",
           phaseDataBlock2[i].chemical_potential[0]);

  if (phaseDataBlock2[i].chemical_potential[1] !=
      phaseDataBlock2[i].chemical_potential[1])
    printf("phaseDataBlock2[i].chemical_potential[1]nan!! "
           "phaseDataBlock2[i].chemical_potential[1]=%.3f\n",
           phaseDataBlock2[i].chemical_potential[1]);
}

template <typename GradientFunc>
__device__ void _ComputeDeltaMassRatio(
    float *deltaMassRatio, const size_t i, const size_t k, const float3 *pos,
    const float *mass, const float *avgDensity, const float eta,
    const float mobilities, const Yang15PhaseDataBlock2 *phaseDataBlock2,
    size_t j, const size_t cellEnd, GradientFunc nablaW) {
  while (j < cellEnd) {

    float3 dp = pos[i] - pos[j];
    *deltaMassRatio += mass[j] / avgDensity[j] * (mobilities + mobilities) *
                       (phaseDataBlock2[i].chemical_potential[k] -
                        phaseDataBlock2[j].chemical_potential[k]) *
                       (dot(dp, nablaW(dp)) / (lengthSquared(dp) + eta * eta));

    ++j;
  }
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void _ComputeMultiSphYang15DeltaMassRatio_CUDA(
    Yang15PhaseDataBlock1 *phaseDataBlock1,
    const Yang15PhaseDataBlock2 *phaseDataBlock2, const float3 *pos,
    const float *mass, const float *avgDensity, const float eta,
    const float mobilities, const size_t num, const size_t phaseNum,
    const size_t *cellStart, const int3 gridSize, Pos2GridXYZ p2xyz,
    GridXYZ2GridHash xyz2hash, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  int3 grid_xyz = p2xyz(pos[i]);

  for (size_t k = 0; k < phaseNum; k++) {
    float delta_mass_ratio = 0.f;

#pragma unroll
    for (size_t m = 0; m < 27; ++m) {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputeDeltaMassRatio(&delta_mass_ratio, i, k, pos, mass, avgDensity,
                             eta, mobilities, phaseDataBlock2,
                             cellStart[hash_idx], cellStart[hash_idx + 1],
                             nablaW);
    }

    phaseDataBlock1[i].delta_mass_ratio[k] = delta_mass_ratio;

    if (phaseDataBlock1[i].delta_mass_ratio[k] !=
        phaseDataBlock1[i].delta_mass_ratio[k])
      printf("phaseDataBlock1[i].delta_mass_ratio[k] nan!! "
             "phaseDataBlock1[i].delta_mass_ratio[k]=%.3f\n",
             phaseDataBlock1[i].delta_mass_ratio[k]);
  }

  return;
}

__global__ void
_CorrectMultiSphYang15MassRatio_CUDA(Yang15PhaseDataBlock1 *phaseDataBlock1,
                                     const size_t num, const size_t phaseNum) {

  size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if (i >= num)
    return;

  float sum_mass_ratio = 0.f;

  for (size_t k = 0; k < phaseNum; k++) {
    phaseDataBlock1[i].last_mass_ratio[k] = phaseDataBlock1[i].mass_ratios[k];
    phaseDataBlock1[i].mass_ratios[k] += phaseDataBlock1[i].delta_mass_ratio[k];

    if (phaseDataBlock1[i].mass_ratios[k] < 0.f)
      phaseDataBlock1[i].mass_ratios[k] = 0.f;

    sum_mass_ratio += phaseDataBlock1[i].mass_ratios[k];
  }

  __syncthreads();

  for (size_t k = 0; k < phaseNum; k++) {
    if (sum_mass_ratio > 0.f)
      phaseDataBlock1[i].mass_ratios[k] /= sum_mass_ratio;
    else
      phaseDataBlock1[i].mass_ratios[k] = phaseDataBlock1[i].last_mass_ratio[k];

    if (phaseDataBlock1[i].mass_ratios[k] != phaseDataBlock1[i].mass_ratios[k])
      printf("phaseDataBlock1[i].mass_ratios[k] nan!! "
             "phaseDataBlock1[i].mass_ratios[k]=%.3f\n",
             phaseDataBlock1[i].mass_ratios[k]);

    // if (phaseDataBlock1[i].delta_mass_ratio[k] > 0.f)
    //     printf("i=%zd,phaseDataBlock1[i].mass_ratios[k]=%.3f\n",
    //            i, phaseDataBlock1[i].mass_ratios[k]);
  }
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void _ComputeMultiSphYang15Lambda_CUDA(
    float *lambda, const Yang15PhaseDataBlock1 *phaseDataBlock1,
    const float *avgDensity, const float3 *pos, const float *mass,
    const size_t num, const size_t *cellStart, const float3 *posB,
    const float *volumeB, const size_t *cellStartB, const int3 gridSize,
    Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  const float constraint =
      avgDensity[i] / phaseDataBlock1[i].aggregate_density - 1.f;

  int3 grid_xyz = p2xyz(pos[i]);

  float3 grad_ci = make_float3(0.f);
  float sum_grad_c2 = 0.f;
  __syncthreads();
#pragma unroll
  for (int m = 0; m < 27; __syncthreads(), ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputePBFGradC(&sum_grad_c2, &grad_ci, i,
                     phaseDataBlock1[i].aggregate_density, pos, mass,
                     cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
    _ComputePBFBoundaryGradC(
        &grad_ci, pos[i], phaseDataBlock1[i].aggregate_density, posB, volumeB,
        cellStartB[hash_idx], cellStartB[hash_idx + 1], nablaW);
  }

  sum_grad_c2 += lengthSquared(grad_ci);
  lambda[i] = -constraint / (sum_grad_c2 + 1000.f);

  if (lambda[i] != lambda[i])
    printf("pbf lambda nan!! lambda[i]=%.3f \n", lambda[i]);

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func,
          typename GradientFunc>
__global__ void _SolveMultiSphYang15DensityConstrain_CUDA(
    float3 *deltaPos, const Yang15PhaseDataBlock1 *phaseDataBlock1,
    const float *lambda, const float3 *pos, const float *mass,
    const float deltaQ, const float corrK, const float corrN, const float dt,
    const size_t num, const size_t *cellStart, const float3 *posB,
    const float *volumeB, const size_t *cellStartB, const int3 gridSize,
    Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, Func W, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  deltaPos[i] = make_float3(0.f);
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

    _ComputePBFDeltaPosRealtime(
        &deltaPos[i], i, phaseDataBlock1[i].aggregate_density, lambda, pos,
        mass, deltaQ, corrK, corrN, dt, cellStart[hash_idx],
        cellStart[hash_idx + 1], W, nablaW);
  }

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func>
__global__ void _ComputeMultiSphYang15ViscosityXSPH_CUDA(
    float3 *acc, const Yang15PhaseDataBlock1 *phaseDataBlock1,
    const float3 *pos, const float3 *vel, const float *avgDensity,
    const float *mass, const float visc, const float boundaryVisc,
    const float dt, const size_t num, const size_t *cellStart,
    const float3 *posB, const float *volumeB, const size_t *cellStartB,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, Func W) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  float3 acc_xsph = make_float3(0.f);

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

    _ComputeViscosityXSPH(&acc_xsph, i, pos, vel, mass, avgDensity, visc,
                          cellStart[hash_idx], cellStart[hash_idx + 1], W);

    _ComputeBoundaryViscosityXSPH(
        &acc_xsph, pos[i], vel[i], avgDensity[i], posB, volumeB,
        phaseDataBlock1[i].aggregate_density, boundaryVisc,
        cellStartB[hash_idx], cellStartB[hash_idx + 1], W);
  }

  if (acc_xsph.x != acc_xsph.x || acc_xsph.y != acc_xsph.y ||
      acc_xsph.z != acc_xsph.z)
    printf("pbf acc_xsph nan!! acc_xsph=%.3f,%.3f,%.3f \n",
           KIRI_EXPAND_FLOAT3(acc_xsph));

  acc[i] += 1.f / dt * acc_xsph;
  return;
}

template <typename GradientFunc>
__device__ void
_ComputeSurfaceTension(float3 *surfaceTension, const size_t i, const size_t k,
                       const float3 *pos, const float *mass,
                       const float *avgDensity, const float eta,
                       const Yang15PhaseDataBlock1 *phaseDataBlock1, size_t j,
                       const size_t cellEnd, GradientFunc nablaW) {
  while (j < cellEnd) {

    float3 dp = pos[i] - pos[j];
    *surfaceTension += mass[j] / avgDensity[j] *
                       (phaseDataBlock1[i].mass_ratios[k] -
                        phaseDataBlock1[j].mass_ratios[k]) *
                       (dot(dp, nablaW(dp)) / (lengthSquared(dp) + eta * eta));

    ++j;
  }
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void _ComputeMultiSphYang15SurfaceTension_CUDA(
    float3 *acc, const Yang15PhaseDataBlock1 *phaseDataBlock1,
    const Yang15PhaseDataBlock2 *phaseDataBlock2, const float3 *pos,
    const float *mass, const float *avgDensity, const float sigma,
    const float eta, const float epsilon, const float dt, const size_t num,
    const size_t phaseNum, const size_t *cellStart, const int3 gridSize,
    Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  int3 grid_xyz = p2xyz(pos[i]);
  float chi = 5.f;
  float3 surface_tension = make_float3(0.f);
  __syncthreads();

  for (size_t k = 0; k < phaseNum; k++) {
    float3 sf_term = make_float3(0.f);
    __syncthreads();
#pragma unroll
    for (size_t m = 0; m < 27; __syncthreads(), ++m) {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputeSurfaceTension(&sf_term, i, k, pos, mass, avgDensity, eta,
                             phaseDataBlock1, cellStart[hash_idx],
                             cellStart[hash_idx + 1], nablaW);
    }
    chi *= phaseDataBlock1[i].mass_ratios[k];
    surface_tension += -6.f * powf(2.f, 0.5f) * epsilon * sf_term *
                       length(phaseDataBlock2[i].mass_ratio_gradient[k]) *
                       phaseDataBlock2[i].mass_ratio_gradient[k];
  }

  acc[i] += dt * sigma / phaseNum * surface_tension * chi / mass[i];

  return;
}

__global__ void _ComputeMultiSphYang15NextTimeStepData_CUDA(
    float3 *color, const Yang15PhaseDataBlock1 *phaseDataBlock1,
    const float3 *color0, const size_t num, const size_t phaseNum) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  float3 col = make_float3(0.f);
  for (size_t k = 0; k < phaseNum; k++)
    col += phaseDataBlock1[i].mass_ratios[k] * color0[k];

  __syncthreads();

  color[i] = col;
  return;
}

} // namespace KIRI
#endif /* _CUDA_MULTISPH_YANG15_SOLVER_GPU_CUH_ */