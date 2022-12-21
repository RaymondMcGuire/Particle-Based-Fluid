/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-12-01 23:00:37
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-12-21 17:07:56
 * @FilePath:
 * \Particle-Based-Fluid-Toolkit\simulator_cuda\src\kiri_pbs_cuda\solver\sph\cuda_iisph_solver.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_pbs_cuda/particle/cuda_iisph_particles.cuh>
#include <kiri_pbs_cuda/solver/cuda_solver_common_gpu.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_iisph_solver.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_iisph_solver_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>

#include <thrust/device_ptr.h>

namespace KIRI {
void CudaIISphSolver::predictVelAdvect(CudaSphParticlesPtr &fluids,
                                       const float dt) {

  auto data = std::dynamic_pointer_cast<CudaIISphParticles>(fluids);
  data->predictVelAdvect(dt);

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaIISphSolver::computeDiiTerm(CudaSphParticlesPtr &fluids,
                                     CudaBoundaryParticlesPtr &boundaries,
                                     const CudaArray<size_t> &cellStart,
                                     const CudaArray<size_t> &boundaryCellStart,
                                     const float rho0, const float3 lowestPoint,
                                     const float kernelRadius,
                                     const int3 gridSize) {
  auto data = std::dynamic_pointer_cast<CudaIISphParticles>(fluids);
  _ComputeDiiTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      data->GetDiiPtr(), data->posPtr(), data->massPtr(), data->densityPtr(),
      rho0, data->size(), cellStart.data(), boundaries->posPtr(),
      boundaries->volumePtr(), boundaryCellStart.data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaIISphSolver::computeAiiTerm(CudaSphParticlesPtr &fluids,
                                     CudaBoundaryParticlesPtr &boundaries,
                                     const CudaArray<size_t> &cellStart,
                                     const CudaArray<size_t> &boundaryCellStart,
                                     const float rho0, const float dt,
                                     const float3 lowestPoint,
                                     const float kernelRadius,
                                     const int3 gridSize) {
  auto data = std::dynamic_pointer_cast<CudaIISphParticles>(fluids);
  _ComputeAiiTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      data->GetAiiPtr(), data->GetDensityAdvPtr(), data->pressurePtr(),
      data->GetDiiPtr(), data->posPtr(), data->velPtr(), data->accPtr(),
      data->massPtr(), data->densityPtr(), data->GetLastPressurePtr(), rho0, dt,
      data->size(), cellStart.data(), boundaries->posPtr(),
      boundaries->volumePtr(), boundaryCellStart.data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

size_t CudaIISphSolver::pressureSolver(
    CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
    const float rho0, const float dt, const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
    const float kernelRadius, const int3 gridSize) {
  auto data = std::dynamic_pointer_cast<CudaIISphParticles>(fluids);
  auto num = data->size();
  auto error = 0.f;
  auto iter = 0;
  auto flag = false;

  while ((!flag || (iter < mMinIter)) && (iter < mMaxIter)) {
    flag = true;

    error = 0.f;

    _ComputeDijPjTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        data->GetDijPjPtr(), data->posPtr(), data->massPtr(),
        data->densityPtr(), data->GetLastPressurePtr(), data->size(),
        cellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        SpikyKernelGrad(kernelRadius));

    _CorrectPressureByJacobi_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        data->pressurePtr(), data->GetLastPressurePtr(),
        data->GetDensityErrorPtr(), data->GetAiiPtr(), data->GetDijPjPtr(),
        data->GetDiiPtr(), data->GetDensityAdvPtr(), data->posPtr(),
        data->massPtr(), data->densityPtr(), rho0, dt, num, cellStart.data(),
        boundaries->posPtr(), boundaries->volumePtr(), boundaryCellStart.data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        SpikyKernelGrad(kernelRadius));

    error = thrust::reduce(
        thrust::device_ptr<float>(data->GetDensityErrorPtr()),
        thrust::device_ptr<float>(data->GetDensityErrorPtr() + num));

    auto eta = mMaxError * rho0 * num;
    flag = flag && (error <= eta);

    iter++;
  }

  // printf("Total Iteration Num=%d, Error=%.6f \n", iter, error);

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();

  return iter;
}

void CudaIISphSolver::computePressureAcceleration(
    CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
    const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float rho0,
    const float3 lowestPoint, const float kernelRadius, const int3 gridSize) {
  auto data = std::dynamic_pointer_cast<CudaIISphParticles>(fluids);
  _ComputePressureAcceleration_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      data->GetPressureAccPtr(), data->posPtr(), data->massPtr(),
      data->densityPtr(), data->pressurePtr(), rho0, data->size(),
      cellStart.data(), boundaries->posPtr(), boundaries->volumePtr(),
      boundaryCellStart.data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaIISphSolver::advect(CudaSphParticlesPtr &fluids, const float dt,
                             const float3 lowestPoint,
                             const float3 highestPoint, const float radius) {
  auto data = std::dynamic_pointer_cast<CudaIISphParticles>(fluids);
  size_t num = data->size();
  data->advect(dt);

  _WorldBoundaryConstrain_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      data->posPtr(), data->velPtr(), num, lowestPoint, highestPoint, radius);

  // ensure reset temp variables
  thrust::fill(thrust::device, data->densityPtr(), data->densityPtr() + num,
               0.f);
  thrust::fill(thrust::device, data->normalPtr(), data->normalPtr() + num,
               make_float3(0.f));
  thrust::fill(thrust::device, data->accPtr(), data->accPtr() + num,
               make_float3(0.f));

  // iisph
  thrust::fill(thrust::device, data->GetAiiPtr(), data->GetAiiPtr() + num, 0.f);
  thrust::fill(thrust::device, data->GetDiiPtr(), data->GetDiiPtr() + num,
               make_float3(0.f));
  thrust::fill(thrust::device, data->GetDijPjPtr(), data->GetDijPjPtr() + num,
               make_float3(0.f));
  thrust::fill(thrust::device, data->GetDensityAdvPtr(),
               data->GetDensityAdvPtr() + num, 0.f);
  thrust::fill(thrust::device, data->GetDensityErrorPtr(),
               data->GetDensityErrorPtr() + num, 0.f);
  thrust::fill(thrust::device, data->GetPressureAccPtr(),
               data->GetPressureAccPtr() + num, make_float3(0.f));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

} // namespace KIRI
