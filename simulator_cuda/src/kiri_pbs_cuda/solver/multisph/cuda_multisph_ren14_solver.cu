/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-11 11:44:38
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-12 23:31:22
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\multisph\cuda_multisph_ren14_solver.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_pbs_cuda/solver/cuda_solver_common_gpu.cuh>
#include <kiri_pbs_cuda/solver/multisph/cuda_multisph_ren14_solver.cuh>
#include <kiri_pbs_cuda/solver/multisph/cuda_multisph_ren14_solver_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>

namespace KIRI
{
  void CudaMultiSphRen14Solver::computeRestMixData(
      CudaMultiSphRen14ParticlesPtr &particles, const size_t phaseNum)
  {
    _ComputeMultiSphRen14MixDensityAndViscosity_CUDA<<<mCudaGridSize,
                                                       KIRI_CUBLOCKSIZE>>>(
        particles->phaseDataBlock1Ptr(), particles->phaseDataBlock2Ptr(),
        particles->restRho0Ptr(), particles->restVisc0Ptr(), particles->size(),
        phaseNum);

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphRen14Solver::computeMixDensity(
      CudaMultiSphRen14ParticlesPtr &particles,
      CudaBoundaryParticlesPtr &boundaries, const size_t phaseNum,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize)
  {
    _ComputeMultiSphRen14MixDensityAndViscosity_CUDA<<<mCudaGridSize,
                                                       KIRI_CUBLOCKSIZE>>>(
        particles->phaseDataBlock1Ptr(), particles->phaseDataBlock2Ptr(),
        particles->restRho0Ptr(), particles->restVisc0Ptr(), particles->size(),
        phaseNum);

    _ComputeMultiSphRen14AverageDensity_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->avgDensityPtr(), particles->posPtr(), particles->mixMassPtr(),
        particles->phaseDataBlock1Ptr(), particles->size(), cellStart.data(),
        boundaries->posPtr(), boundaries->volumePtr(), boundaryCellStart.data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize), Poly6Kernel(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphRen14Solver::computeMixPressure(
      CudaMultiSphRen14ParticlesPtr &particles, const size_t phaseNum,
      const bool miscible, const float stiff)
  {

    _ComputeMultiSphRen14Pressure_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->phaseDataBlock1Ptr(), particles->phaseDataBlock2Ptr(),
        particles->avgDensityPtr(), particles->size(), phaseNum, miscible, stiff,
        1.f);

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphRen14Solver::computeGradientTerm(
      CudaMultiSphRen14ParticlesPtr &particles, const size_t phaseNum,
      const bool miscible, const CudaArray<size_t> &cellStart,
      const float3 lowestPoint, const float kernelRadius, const int3 gridSize)
  {

    _ComputeMultiSphRen14GradientTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->phaseDataBlock1Ptr(), particles->phaseDataBlock2Ptr(),
        particles->posPtr(), particles->mixMassPtr(), particles->avgDensityPtr(),
        particles->size(), phaseNum, miscible, cellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphRen14Solver::computeDriftVelocities(
      CudaMultiSphRen14ParticlesPtr &particles, const size_t phaseNum,
      const bool miscible, const float tou, const float sigma,
      const float3 gravity)
  {

    _ComputeMultiSphRen14DriftVelocities_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->phaseDataBlock1Ptr(), particles->phaseDataBlock2Ptr(),
        particles->accPtr(), particles->restRho0Ptr(), particles->size(),
        phaseNum, miscible, tou, sigma, gravity);

    // _UpdateMultiSphRen14VelocityByDriftVelocity_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
    //     particles->velPtr(), particles->phaseDataBlock1Ptr(),
    //     particles->phaseDataBlock2Ptr(), particles->size(), phaseNum,
    //     particles->restRho0Ptr());

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphRen14Solver::computeDeltaVolumeFraction(
      CudaMultiSphRen14ParticlesPtr &particles, const size_t phaseNum,
      const bool miscible, const CudaArray<size_t> &cellStart,
      const float3 lowestPoint, const float kernelRadius, const int3 gridSize)
  {

    _ComputeMultiSphRen14DeltaVolumeFraction_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->phaseDataBlock2Ptr(), particles->posPtr(),
        particles->mixMassPtr(), particles->velPtr(), particles->avgDensityPtr(),
        particles->size(), phaseNum, miscible, cellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphRen14Solver::correctVolumeFraction(
      CudaMultiSphRen14ParticlesPtr &particles,
      const size_t phaseNum,
      const bool miscible,
      const float stiff,
      const float dt)
  {

    _CorrectMultiSphRen14VolumeFraction_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->phaseDataBlock1Ptr(),
        particles->phaseDataBlock2Ptr(),
        particles->restRho0Ptr(),
        particles->size(),
        phaseNum,
        miscible,
        stiff,
        mVolumeFractionChangeSpeed,
        dt);

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphRen14Solver::computeMultiSphAcc(
      CudaMultiSphRen14ParticlesPtr &particles,
      CudaBoundaryParticlesPtr &boundaries, const size_t phaseNum,
      const float3 gravity, const float soundSpeed, const float bnu,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize)
  {

    _ComputeMultiSphRen14Acc_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->accPtr(), particles->posPtr(), particles->mixMassPtr(),
        particles->velPtr(), particles->avgDensityPtr(),
        particles->phaseDataBlock1Ptr(), particles->phaseDataBlock2Ptr(),
        particles->restRho0Ptr(), gravity, kernelRadius, soundSpeed, bnu,
        particles->size(), phaseNum, cellStart.data(), boundaries->posPtr(),
        boundaries->volumePtr(), boundaryCellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphRen14Solver::computeNextTimeStepData(
      CudaMultiSphRen14ParticlesPtr &particles, const size_t phaseNum, const bool miscible)
  {

    _ComputeMultiSphRen14PhaseColorAndMass_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->mixMassPtr(), particles->colorPtr(),
        particles->phaseDataBlock1Ptr(), particles->phaseDataBlock2Ptr(),
        particles->size(), phaseNum, particles->restMass0Ptr(), miscible,
        particles->restColor0Ptr());

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphRen14Solver::advect(CudaMultiSphRen14ParticlesPtr &particles,
                                       const float dt, const float3 lowestPoint,
                                       const float3 highestPoint,
                                       const float radius)
  {

    size_t num = particles->size();
    particles->advect(dt);
    _WorldBoundaryConstrain_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->posPtr(), particles->velPtr(), num, lowestPoint, highestPoint,
        radius);

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

} // namespace KIRI
