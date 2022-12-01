/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2021-03-29 12:45:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-05-18 17:49:41
 * @FilePath:
 * \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\sph\cuda_wcsph_solver.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_pbs_cuda/solver/sph/cuda_wcsph_solver.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_wcsph_solver_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>

#include <thrust/device_ptr.h>

namespace KIRI
{

  void CudaWCSphSolver::computeSubTimeStepsByCFL(CudaSphParticlesPtr &fluids,
                                                 const float restMass,
                                                 const float kernelRadius,
                                                 float timeIntervalInSeconds)
  {

    auto accArray = thrust::device_pointer_cast(fluids->accPtr());
    float3 maxAcc =
        *(thrust::max_element(accArray, accArray + fluids->size(),
                              ThrustHelper::CompareLengthCuda<float3>()));

    float maxForceMagnitude = length(maxAcc) * restMass;
    float timeStepLimitBySpeed =
        mTimeStepLimitBySpeedFactor * kernelRadius / mSpeedOfSound;
    float timeStepLimitByForce =
        mTimeStepLimitByForceFactor *
        std::sqrt(kernelRadius * restMass / maxForceMagnitude);
    float desiredTimeStep = mTimeStepLimitScale *
                            std::min(timeStepLimitBySpeed, timeStepLimitByForce);

    mNumOfSubTimeSteps =
        static_cast<int>(std::ceil(timeIntervalInSeconds / desiredTimeStep));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaWCSphSolver::computeDensity(
      CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
      const float rho0, const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize)
  {
    if (bCubicKernel)
      _ComputeDensity_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
          fluids->densityPtr(), fluids->posPtr(), fluids->massPtr(), rho0,
          fluids->size(), cellStart.data(), boundaries->posPtr(),
          boundaries->volumePtr(), boundaryCellStart.data(), gridSize,
          ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
          ThrustHelper::GridXYZ2GridHash(gridSize), CubicKernel(kernelRadius));
    else
      _ComputeDensity_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
          fluids->densityPtr(), fluids->posPtr(), fluids->massPtr(), rho0,
          fluids->size(), cellStart.data(), boundaries->posPtr(),
          boundaries->volumePtr(), boundaryCellStart.data(), gridSize,
          ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
          ThrustHelper::GridXYZ2GridHash(gridSize), Poly6Kernel(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaWCSphSolver::computeNablaTerm(
      CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize, const float rho0,
      const float stiff)
  {

    _ComputePressureByTait_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->densityPtr(), fluids->pressurePtr(), fluids->size(), rho0, stiff,
        mNegativeScale);
    if (bCubicKernel)
      _ComputeNablaTermConstrain_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
          fluids->posPtr(), fluids->accPtr(), fluids->massPtr(),
          fluids->densityPtr(), fluids->pressurePtr(), rho0, fluids->size(),
          cellStart.data(), boundaries->posPtr(), boundaries->volumePtr(),
          boundaryCellStart.data(), gridSize,
          ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
          ThrustHelper::GridXYZ2GridHash(gridSize),
          CubicKernelGrad(kernelRadius));
    else
      _ComputeNablaTermConstrain_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
          fluids->posPtr(), fluids->accPtr(), fluids->massPtr(),
          fluids->densityPtr(), fluids->pressurePtr(), rho0, fluids->size(),
          cellStart.data(), boundaries->posPtr(), boundaries->volumePtr(),
          boundaryCellStart.data(), gridSize,
          ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
          ThrustHelper::GridXYZ2GridHash(gridSize),
          SpikyKernelGrad(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaWCSphSolver::computeViscosityTerm(
      CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float rho0,
      const float visc, const float bnu, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize)
  {
    if (bCubicKernel)
      _ComputeViscosityTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
          fluids->accPtr(), fluids->posPtr(), fluids->velPtr(), fluids->massPtr(),
          fluids->densityPtr(), rho0, visc, bnu, fluids->size(), cellStart.data(),
          boundaries->posPtr(), boundaries->volumePtr(),
          boundaryCellStart.data(), gridSize,
          ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
          ThrustHelper::GridXYZ2GridHash(gridSize), CubicKernelGrad(kernelRadius),
          ViscosityKernelLaplacian(kernelRadius));
    else
      _ComputeViscosityTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
          fluids->accPtr(), fluids->posPtr(), fluids->velPtr(), fluids->massPtr(),
          fluids->densityPtr(), rho0, visc, bnu, fluids->size(), cellStart.data(),
          boundaries->posPtr(), boundaries->volumePtr(),
          boundaryCellStart.data(), gridSize,
          ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
          ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius),
          SpikyKernelLaplacian(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaWCSphSolver::computeArtificialViscosityTerm(
      CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float rho0,
      const float nu, const float bnu, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize)
  {
    if (bCubicKernel)
      _ComputeArtificialViscosityTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
          fluids->accPtr(), fluids->posPtr(), fluids->velPtr(), fluids->massPtr(),
          fluids->densityPtr(), rho0, nu, bnu, kernelRadius, fluids->size(), cellStart.data(),
          boundaries->posPtr(), boundaries->volumePtr(),
          boundaryCellStart.data(), gridSize,
          ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
          ThrustHelper::GridXYZ2GridHash(gridSize),
          CubicKernelGrad(kernelRadius));
    else
      _ComputeArtificialViscosityTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
          fluids->accPtr(), fluids->posPtr(), fluids->velPtr(), fluids->massPtr(),
          fluids->densityPtr(), rho0, nu, bnu, kernelRadius, fluids->size(), cellStart.data(),
          boundaries->posPtr(), boundaries->volumePtr(),
          boundaryCellStart.data(), gridSize,
          ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
          ThrustHelper::GridXYZ2GridHash(gridSize),
          SpikyKernelGrad(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }
} // namespace KIRI
