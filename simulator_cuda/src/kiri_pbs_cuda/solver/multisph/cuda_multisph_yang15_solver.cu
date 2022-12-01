/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-24 14:09:09
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-24 14:49:27
 * @FilePath:
 * \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\multisph\cuda_multisph_yang15_solver.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_pbs_cuda/solver/cuda_solver_common_gpu.cuh>
#include <kiri_pbs_cuda/solver/multisph/cuda_multisph_yang15_solver.cuh>
#include <kiri_pbs_cuda/solver/multisph/cuda_multisph_yang15_solver_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>

namespace KIRI {

void CudaMultiSphYang15Solver::extraForces(
    CudaMultiSphYang15ParticlesPtr &particles, const float3 gravity) {

  thrust::transform(thrust::device, particles->accPtr(),
                    particles->accPtr() + particles->size(),
                    particles->accPtr(), ThrustHelper::Plus<float3>(gravity));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaMultiSphYang15Solver::computeAggregateDensity(
    CudaMultiSphYang15ParticlesPtr &particles,
    CudaBoundaryParticlesPtr &boundaries, const size_t phaseNum,
    const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
    const float kernelRadius, const int3 gridSize) {
  _ComputeMultiSphYang15AggregateData_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      particles->mixMassPtr(), particles->phaseDataBlock1Ptr(),
      particles->restRho0Ptr(), particles->restMass0Ptr(), particles->size(),
      phaseNum);

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();

  _ComputeMultiSphYang15AggregateDensity_CUDA<<<mCudaGridSize,
                                                KIRI_CUBLOCKSIZE>>>(
      particles->avgDensityPtr(), particles->densityErrorPtr(),
      particles->posPtr(), particles->mixMassPtr(),
      particles->phaseDataBlock1Ptr(), particles->size(), cellStart.data(),
      boundaries->posPtr(), boundaries->volumePtr(), boundaryCellStart.data(),
      gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), Poly6Kernel(kernelRadius));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaMultiSphYang15Solver::computeNSCHModel(
    CudaMultiSphYang15ParticlesPtr &particles, const size_t phaseNum,
    const float eta, const float mobilities, const float alpha, const float s1,
    const float s2, const float epsilon, const CudaArray<size_t> &cellStart,
    const float3 lowestPoint, const float kernelRadius, const int3 gridSize) {
  _ComputeMultiSphYang15MassRatioTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      particles->phaseDataBlock2Ptr(), particles->phaseDataBlock1Ptr(),
      particles->posPtr(), particles->mixMassPtr(), particles->avgDensityPtr(),
      eta, particles->size(), phaseNum, cellStart.data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();

  _ComputeMultiSphYang15ChemicalPotential_CUDA<<<mCudaGridSize,
                                                 KIRI_CUBLOCKSIZE>>>(
      particles->phaseDataBlock2Ptr(), particles->phaseDataBlock1Ptr(),
      particles->size(), alpha, s1, s2, epsilon);

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();

  _ComputeMultiSphYang15DeltaMassRatio_CUDA<<<mCudaGridSize,
                                              KIRI_CUBLOCKSIZE>>>(
      particles->phaseDataBlock1Ptr(), particles->phaseDataBlock2Ptr(),
      particles->posPtr(), particles->mixMassPtr(), particles->avgDensityPtr(),
      eta, mobilities, particles->size(), phaseNum, cellStart.data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();

  _CorrectMultiSphYang15MassRatio_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      particles->phaseDataBlock1Ptr(), particles->size(), phaseNum);

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaMultiSphYang15Solver::computeLagrangeMultiplier(
    CudaMultiSphYang15ParticlesPtr &particles,
    CudaBoundaryParticlesPtr &boundaries, const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
    const float kernelRadius, const int3 gridSize) {
  // NOTE reset density error value in device
  thrust::fill(thrust::device, particles->densityErrorPtr(),
               particles->densityErrorPtr() + 1, 0.f);

  _ComputeMultiSphYang15AggregateDensity_CUDA<<<mCudaGridSize,
                                                KIRI_CUBLOCKSIZE>>>(
      particles->avgDensityPtr(), particles->densityErrorPtr(),
      particles->posPtr(), particles->mixMassPtr(),
      particles->phaseDataBlock1Ptr(), particles->size(), cellStart.data(),
      boundaries->posPtr(), boundaries->volumePtr(), boundaryCellStart.data(),
      gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), Poly6Kernel(kernelRadius));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();

  // NOTE record average density error
  mAvgDensityError = particles->densityErrorSum() / particles->size();

  _ComputeMultiSphYang15Lambda_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      particles->lambdaPtr(), particles->phaseDataBlock1Ptr(),
      particles->avgDensityPtr(), particles->posPtr(), particles->mixMassPtr(),
      particles->size(), cellStart.data(), boundaries->posPtr(),
      boundaries->volumePtr(), boundaryCellStart.data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaMultiSphYang15Solver::solveDensityConstrain(
    CudaMultiSphYang15ParticlesPtr &particles,
    CudaBoundaryParticlesPtr &boundaries, const float dt,
    const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
    const float kernelRadius, const int3 gridSize) {
  _SolveMultiSphYang15DensityConstrain_CUDA<<<mCudaGridSize,
                                              KIRI_CUBLOCKSIZE>>>(
      particles->deltaPosPtr(), particles->phaseDataBlock1Ptr(),
      particles->lambdaPtr(), particles->posPtr(), particles->mixMassPtr(),
      0.3f * kernelRadius, 0.05f, 4.f, dt, particles->size(), cellStart.data(),
      boundaries->posPtr(), boundaries->volumePtr(), boundaryCellStart.data(),
      gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), Poly6Kernel(kernelRadius),
      SpikyKernelGrad(kernelRadius));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaMultiSphYang15Solver::constaintProjection(
    CudaMultiSphYang15ParticlesPtr &particles,
    CudaBoundaryParticlesPtr &boundaries, const float dt,
    const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
    const float kernelRadius, const int3 gridSize) {

  computeLagrangeMultiplier(particles, boundaries, cellStart, boundaryCellStart,
                            lowestPoint, kernelRadius, gridSize);

  solveDensityConstrain(particles, boundaries, dt, cellStart, boundaryCellStart,
                        lowestPoint, kernelRadius, gridSize);

  particles->correctPos();
}

void CudaMultiSphYang15Solver::computeViscosityXSPH(
    CudaMultiSphYang15ParticlesPtr &particles,
    CudaBoundaryParticlesPtr &boundaries, const float visc,
    const float boundaryVisc, const float dt,
    const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
    const float kernelRadius, const int3 gridSize) {

  _ComputeMultiSphYang15ViscosityXSPH_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      particles->accPtr(), particles->phaseDataBlock1Ptr(), particles->posPtr(),
      particles->velPtr(), particles->avgDensityPtr(), particles->mixMassPtr(),
      visc, boundaryVisc, dt, particles->size(), cellStart.data(),
      boundaries->posPtr(), boundaries->volumePtr(), boundaryCellStart.data(),
      gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), Poly6Kernel(kernelRadius));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaMultiSphYang15Solver::computeSurfaceTension(
    CudaMultiSphYang15ParticlesPtr &particles, const float sigma,
    const float eta, const float epsilon, const float dt, const size_t phaseNum,
    const CudaArray<size_t> &cellStart, const float3 lowestPoint,
    const float kernelRadius, const int3 gridSize) {
  _ComputeMultiSphYang15SurfaceTension_CUDA<<<mCudaGridSize,
                                              KIRI_CUBLOCKSIZE>>>(
      particles->accPtr(), particles->phaseDataBlock1Ptr(),
      particles->phaseDataBlock2Ptr(), particles->posPtr(),
      particles->mixMassPtr(), particles->avgDensityPtr(), sigma, eta, epsilon,
      dt, particles->size(), phaseNum, cellStart.data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaMultiSphYang15Solver::computeNextTimeStepData(
    CudaMultiSphYang15ParticlesPtr &particles, const size_t phaseNum) {

  _ComputeMultiSphYang15NextTimeStepData_CUDA<<<mCudaGridSize,
                                                KIRI_CUBLOCKSIZE>>>(
      particles->colorPtr(), particles->phaseDataBlock1Ptr(),
      particles->restColor0Ptr(), particles->size(), phaseNum);

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaMultiSphYang15Solver::advect(CudaMultiSphYang15ParticlesPtr &particles,
                                      const float dt, const float3 lowestPoint,
                                      const float3 highestPoint,
                                      const float radius) {
  size_t num = particles->size();
  particles->advect(dt);
  _WorldBoundaryConstrain_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      particles->posPtr(), particles->velPtr(), num, lowestPoint, highestPoint,
      radius);
  thrust::fill(thrust::device, particles->avgDensityPtr(),
      particles->avgDensityPtr() + num, 0.f);
  thrust::fill(thrust::device, particles->accPtr(), particles->accPtr() + num,
               make_float3(0.f));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

} // namespace KIRI
