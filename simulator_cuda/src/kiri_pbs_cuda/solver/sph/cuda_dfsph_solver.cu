/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-12-01 23:00:37
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-12-21 17:07:48
 * @FilePath:
 * \Particle-Based-Fluid-Toolkit\simulator_cuda\src\kiri_pbs_cuda\solver\sph\cuda_dfsph_solver.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_pbs_cuda/particle/cuda_dfsph_particles.cuh>
#include <kiri_pbs_cuda/solver/cuda_solver_common_gpu.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_dfsph_solver.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_dfsph_solver_gpu.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver_common_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>

#include <thrust/device_ptr.h>

namespace KIRI {

void CudaDFSphSolver::computeDensity(
    CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
    const float rho0, const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
    const float kernelRadius, const int3 gridSize) {
  auto data = std::dynamic_pointer_cast<CudaDFSphParticles>(fluids);

  _ComputeDensity_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      data->densityPtr(), data->posPtr(), data->massPtr(), rho0, data->size(),
      cellStart.data(), boundaries->posPtr(), boundaries->volumePtr(),
      boundaryCellStart.data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), CubicKernel(kernelRadius));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaDFSphSolver::velAdvect(CudaSphParticlesPtr &fluids) {
  auto data = std::dynamic_pointer_cast<CudaDFSphParticles>(fluids);
  data->velAdvect(mDt);
}

void CudaDFSphSolver::computeTimeStepsByCFL(CudaSphParticlesPtr &fluids,
                                            const float particleRadius,
                                            const float timeIntervalInSeconds) {

  auto data = std::dynamic_pointer_cast<CudaDFSphParticles>(fluids);
  _ComputeVelMag_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      data->velMagPtr(), data->velPtr(), data->accPtr(), mDt, data->size());

  auto vel_mag_array = thrust::device_pointer_cast(data->velMagPtr());
  float max_vel_mag =
      *(thrust::max_element(vel_mag_array, vel_mag_array + data->size()));

  auto diam = 2.f * particleRadius;
  mDt = CFL_FACTOR * 0.4f * (diam / sqrt(max_vel_mag));
  mDt = max(mDt, CFL_MIN_TIMESTEP_SIZE);
  mDt = min(mDt, CFL_MAX_TIMESTEP_SIZE);

  mNumOfSubTimeSteps = static_cast<int>(std::ceil(timeIntervalInSeconds / mDt));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaDFSphSolver::computeAlpha(
    CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
    const float rho0, const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
    const float kernelRadius, const int3 gridSize) {
  auto data = std::dynamic_pointer_cast<CudaDFSphParticles>(fluids);
  _ComputeAlpha_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      data->alphaPtr(), data->posPtr(), data->massPtr(), data->densityPtr(),
      rho0, data->size(), cellStart.data(), boundaries->posPtr(),
      boundaries->volumePtr(), boundaryCellStart.data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), CubicKernelGrad(kernelRadius));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

size_t CudaDFSphSolver::divergenceSolver(
    CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
    const float rho0, const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
    const float kernelRadius, const int3 gridSize) {
  auto data = std::dynamic_pointer_cast<CudaDFSphParticles>(fluids);
  auto num = data->size();

  // Compute velocity of density change
  _ComputeDivgenceError_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      data->stiffPtr(), data->densityErrorPtr(), data->alphaPtr(),
      data->posPtr(), data->velPtr(), data->massPtr(), data->densityPtr(), rho0,
      mDt, num, cellStart.data(), boundaries->posPtr(), boundaries->volumePtr(),
      boundaryCellStart.data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), CubicKernelGrad(kernelRadius));

  auto iter = 0;
  auto total_error = std::numeric_limits<float>::max();

  while ((total_error > mDivergenceErrorThreshold * num * rho0 ||
          (iter < mDivergenceMinIter)) &&
         (iter < mDivergenceMaxIter)) {

    _CorrectDivergenceByJacobi_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        data->velPtr(), data->stiffPtr(), data->posPtr(), data->massPtr(), rho0,
        num, cellStart.data(), boundaries->posPtr(), boundaries->volumePtr(),
        boundaryCellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        CubicKernelGrad(kernelRadius));

    _ComputeDivgenceError_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        data->stiffPtr(), data->densityErrorPtr(), data->alphaPtr(),
        data->posPtr(), data->velPtr(), data->massPtr(), data->densityPtr(),
        rho0, mDt, num, cellStart.data(), boundaries->posPtr(),
        boundaries->volumePtr(), boundaryCellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        CubicKernelGrad(kernelRadius));

    iter++;

    total_error =
        thrust::reduce(thrust::device_ptr<float>(data->densityErrorPtr()),
                       thrust::device_ptr<float>(data->densityErrorPtr() + num),
                       0.f, ThrustHelper::AbsPlus<float>());
  }

  // printf("divergence iter=%d, total_error=%.6f \n", iter,
  //        total_error);
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();

  return iter;
}

size_t CudaDFSphSolver::pressureSolver(
    CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
    const float rho0, const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
    const float kernelRadius, const int3 gridSize) {
  auto data = std::dynamic_pointer_cast<CudaDFSphParticles>(fluids);
  auto num = data->size();

  // use warm stiff
  _CorrectPressureByJacobi_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      data->velPtr(), data->warmStiffPtr(), data->posPtr(), data->massPtr(),
      rho0, mDt, num, cellStart.data(), boundaries->posPtr(),
      boundaries->volumePtr(), boundaryCellStart.data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), CubicKernelGrad(kernelRadius));

  _ComputeDensityError_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      data->densityErrorPtr(), data->stiffPtr(), data->alphaPtr(),
      data->posPtr(), data->velPtr(), data->massPtr(), data->densityPtr(), rho0,
      mDt, num, cellStart.data(), boundaries->posPtr(), boundaries->volumePtr(),
      boundaryCellStart.data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), CubicKernelGrad(kernelRadius));

  // reset warm stiffness
  KIRI_CUCALL(cudaMemcpy(data->warmStiffPtr(), data->stiffPtr(),
                         sizeof(float) * num, cudaMemcpyDeviceToDevice));

  auto iter = 0;
  auto total_error = std::numeric_limits<float>::max();

  while ((total_error > mPressureErrorThreshold * num * rho0 ||
          (iter < mPressureMinIter)) &&
         (iter < mPressureMaxIter)) {

    _CorrectPressureByJacobi_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        data->velPtr(), data->stiffPtr(), data->posPtr(), data->massPtr(), rho0,
        mDt, num, cellStart.data(), boundaries->posPtr(),
        boundaries->volumePtr(), boundaryCellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        CubicKernelGrad(kernelRadius));

    _ComputeDensityError_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        data->densityErrorPtr(), data->stiffPtr(), data->alphaPtr(),
        data->posPtr(), data->velPtr(), data->massPtr(), data->densityPtr(),
        rho0, mDt, num, cellStart.data(), boundaries->posPtr(),
        boundaries->volumePtr(), boundaryCellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        CubicKernelGrad(kernelRadius));

    thrust::transform(thrust::device, data->warmStiffPtr(),
                      data->warmStiffPtr() + num, data->stiffPtr(),
                      data->warmStiffPtr(), thrust::plus<float>());
    iter++;

    if (iter >= mPressureMinIter) {
      total_error = thrust::reduce(
          thrust::device_ptr<float>(data->densityErrorPtr()),
          thrust::device_ptr<float>(data->densityErrorPtr() + num), 0.f,
          ThrustHelper::AbsPlus<float>());
    }
  }

  //   printf("Total Iteration Num=%d; Total Error=%.6f; Threshold=%.6f \n",
  //   iter,
  //          total_error, mPressureErrorThreshold * num * rho0);

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();

  return iter;
}

void CudaDFSphSolver::advect(CudaSphParticlesPtr &fluids, const float dt,
                             const float3 lowestPoint,
                             const float3 highestPoint, const float radius) {
  auto data = std::dynamic_pointer_cast<CudaDFSphParticles>(fluids);
  size_t num = data->size();
  data->advect(dt);

  _WorldBoundaryConstrain_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      data->posPtr(), data->velPtr(), num, lowestPoint, highestPoint, radius);

  thrust::fill(thrust::device, data->normalPtr(), data->normalPtr() + num,
               make_float3(0.f));
  thrust::fill(thrust::device, data->accPtr(), data->accPtr() + num,
               make_float3(0.f));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

} // namespace KIRI
