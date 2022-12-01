/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-12 14:48:13
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-20 10:52:37
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\sph\cuda_pbf_solver.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_pbs_cuda/solver/sph/cuda_pbf_solver.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_pbf_solver_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
#include <thrust/device_ptr.h>

namespace KIRI
{

  void CudaPBFSolver::computeLagrangeMultiplier(
      CudaPBFParticlesPtr &fluids,
      CudaBoundaryParticlesPtr &boundaries,
      const float rho0,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart,
      const float3 lowestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {

    // NOTE reset density error value in device
    thrust::fill(thrust::device, fluids->densityErrorPtr(),
                 fluids->densityErrorPtr() + 1, 0.f);

    _ComputePBFDensity_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->densityPtr(), fluids->densityErrorPtr(), fluids->posPtr(),
        fluids->massPtr(), rho0, fluids->size(), cellStart.data(),
        boundaries->posPtr(), boundaries->volumePtr(),
        boundaryCellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize), Poly6Kernel(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();

    // NOTE record average density error
    mAvgDensityError = fluids->densityErrorSum() / fluids->size();
    // printf("after total density_error=%.3f, size=%d \n", mAvgDensityError,
    //        fluids->size());
    if (mIncompressiable)
      _ComputePBFLambdaIncompressible_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
          fluids->lambdaPtr(),
          fluids->densityPtr(),
          fluids->posPtr(),
          fluids->massPtr(),
          rho0,
          fluids->size(),
          cellStart.data(),
          boundaries->posPtr(),
          boundaries->volumePtr(),
          boundaryCellStart.data(),
          gridSize,
          ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
          ThrustHelper::GridXYZ2GridHash(gridSize),
          SpikyKernelGrad(kernelRadius));
    else
      _ComputePBFLambdaRealtime_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
          fluids->lambdaPtr(),
          fluids->densityPtr(),
          fluids->posPtr(),
          fluids->massPtr(),
          rho0,
          mLambdaEps,
          fluids->size(),
          cellStart.data(),
          boundaries->posPtr(),
          boundaries->volumePtr(),
          boundaryCellStart.data(),
          gridSize,
          ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
          ThrustHelper::GridXYZ2GridHash(gridSize),
          SpikyKernelGrad(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaPBFSolver::solveDensityConstrain(
      CudaPBFParticlesPtr &fluids,
      CudaBoundaryParticlesPtr &boundaries,
      const float rho0,
      const float dt,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart,
      const float3 lowestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {
    if (mIncompressiable)
      _SolvePBFDensityConstrainIncompressible_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
          fluids->deltaPosPtr(),
          fluids->lambdaPtr(),
          fluids->densityPtr(),
          fluids->posPtr(),
          fluids->massPtr(),
          rho0,
          fluids->size(),
          cellStart.data(),
          boundaries->posPtr(),
          boundaries->volumePtr(),
          boundaryCellStart.data(), gridSize,
          ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
          ThrustHelper::GridXYZ2GridHash(gridSize),
          SpikyKernelGrad(kernelRadius));
    else
      _SolvePBFDensityConstrainRealtime_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
          fluids->deltaPosPtr(),
          fluids->lambdaPtr(),
          fluids->densityPtr(),
          fluids->posPtr(),
          fluids->massPtr(),
          rho0,
          mDeltaQ * kernelRadius,
          mCorrK,
          mCorrN,
          dt,
          fluids->size(),
          cellStart.data(),
          boundaries->posPtr(),
          boundaries->volumePtr(),
          boundaryCellStart.data(), gridSize,
          ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
          ThrustHelper::GridXYZ2GridHash(gridSize),
          Poly6Kernel(kernelRadius),
          SpikyKernelGrad(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaPBFSolver::constaintProjection(
      CudaPBFParticlesPtr &fluids,
      CudaBoundaryParticlesPtr &boundaries,
      const float rho0,
      const float dt,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart,
      const float3 lowestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {

    computeLagrangeMultiplier(
        fluids,
        boundaries,
        rho0,
        cellStart,
        boundaryCellStart,
        lowestPoint,
        kernelRadius,
        gridSize);

    solveDensityConstrain(
        fluids,
        boundaries,
        rho0,
        dt,
        cellStart,
        boundaryCellStart,
        lowestPoint,
        kernelRadius,
        gridSize);

    fluids->correctPos();
  }

  void CudaPBFSolver::computeViscosityXSPH(
      CudaPBFParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
      const float rho0, const float visc, const float boundaryVisc,
      const float dt, const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize)
  {

    _ComputeViscosityXSPH_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->accPtr(),
        fluids->posPtr(),
        fluids->velPtr(),
        fluids->densityPtr(),
        fluids->massPtr(),
        rho0,
        visc,
        boundaryVisc,
        dt,
        fluids->size(),
        cellStart.data(),
        boundaries->posPtr(),
        boundaries->volumePtr(),
        boundaryCellStart.data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        Poly6Kernel(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaPBFSolver::computeVorticityConfinement(
      CudaPBFParticlesPtr &fluids,
      const float vorticityCoeff,
      const CudaArray<size_t> &cellStart,
      const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize)
  {

    _ComputeVorticityOmega_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->omegaPtr(),
        fluids->normOmegaPtr(),
        fluids->posPtr(),
        fluids->velPtr(),
        fluids->densityPtr(),
        fluids->massPtr(),
        fluids->size(),
        cellStart.data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        SpikyKernelGrad(kernelRadius));

    _ComputeVorticityConfinement_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->accPtr(),
        fluids->omegaPtr(),
        fluids->normOmegaPtr(),
        fluids->posPtr(),
        fluids->densityPtr(),
        fluids->massPtr(),
        vorticityCoeff,
        fluids->size(),
        cellStart.data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        SpikyKernelGrad(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

} // namespace KIRI
