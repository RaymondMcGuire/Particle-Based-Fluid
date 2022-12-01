/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-05-11 15:18:08
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-05-18 17:49:15
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\sph\cuda_sph_solver.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_pbs_cuda/solver/cuda_solver_common_gpu.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_akinci13_gpu.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>

namespace KIRI
{
  void CudaSphSolver::computeDensity(
      CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
      const float rho0, const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize)
  {
    _ComputeDensity_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->densityPtr(), fluids->posPtr(), fluids->massPtr(), rho0,
        fluids->size(), cellStart.data(), boundaries->posPtr(),
        boundaries->volumePtr(), boundaryCellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize), Poly6Kernel(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSolver::computeNablaTerm(CudaSphParticlesPtr &fluids,
                                       CudaBoundaryParticlesPtr &boundaries,
                                       const CudaArray<size_t> &cellStart,
                                       const CudaArray<size_t> &boundaryCellStart,
                                       const float3 lowestPoint,
                                       const float kernelRadius,
                                       const int3 gridSize, const float rho0,
                                       const float stiff)
  {

    _ComputePressure_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->pressurePtr(), fluids->densityPtr(), fluids->size(), rho0, stiff,
        1.f);

    _ComputeNablaTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->posPtr(), fluids->accPtr(), fluids->massPtr(),
        fluids->densityPtr(), fluids->pressurePtr(), rho0, fluids->size(),
        cellStart.data(), boundaries->posPtr(), boundaries->volumePtr(),
        boundaryCellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSolver::computeViscosityTerm(
      CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float rho0,
      const float visc, const float bnu, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize)
  {
    _ComputeViscosityTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->accPtr(), fluids->posPtr(), fluids->velPtr(), fluids->massPtr(),
        fluids->densityPtr(), rho0, visc, bnu, fluids->size(), cellStart.data(),
        boundaries->posPtr(), boundaries->volumePtr(),
        boundaryCellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius),
        ViscosityKernelLaplacian(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSolver::computeArtificialViscosityTerm(
      CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float rho0,
      const float nu, const float bnu, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize)
  {
    _ComputeArtificialViscosityTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->accPtr(), fluids->posPtr(), fluids->velPtr(), fluids->massPtr(),
        fluids->densityPtr(), rho0, nu, bnu, kernelRadius, fluids->size(), cellStart.data(),
        boundaries->posPtr(), boundaries->volumePtr(),
        boundaryCellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSolver::computeAkinci13Normal(CudaSphParticlesPtr &fluids,
                                            const CudaArray<size_t> &cellStart,
                                            const float3 lowestPoint,
                                            const float kernelRadius,
                                            const int3 gridSize)
  {
    _ComputeNormal_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->posPtr(), fluids->normalPtr(), fluids->massPtr(),
        fluids->densityPtr(), kernelRadius, fluids->size(), cellStart.data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSolver::computeAkinci13Term(
      CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float rho0,
      const float beta, const float gamma, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize)
  {
    _ComputeAkinci13Term_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->posPtr(), fluids->accPtr(), fluids->massPtr(),
        fluids->densityPtr(), fluids->normalPtr(), rho0, gamma, beta,
        fluids->size(), cellStart.data(), boundaries->posPtr(),
        boundaries->volumePtr(), boundaryCellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        SurfaceTensionAkinci13(kernelRadius), AdhesionAkinci13(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSolver::advect(CudaSphParticlesPtr &fluids, const float dt,
                             const float3 lowestPoint, const float3 highestPoint,
                             const float radius)
  {
    size_t num = fluids->size();
    fluids->advect(dt);
    _WorldBoundaryConstrain_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->posPtr(), fluids->velPtr(), num, lowestPoint, highestPoint,
        radius);

    thrust::fill(thrust::device, fluids->densityPtr(), fluids->densityPtr() + num,
                 0.f);
    thrust::fill(thrust::device, fluids->normalPtr(), fluids->normalPtr() + num,
                 make_float3(0.f));
    thrust::fill(thrust::device, fluids->accPtr(), fluids->accPtr() + num,
                 make_float3(0.f));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSolver::extraForces(CudaSphParticlesPtr &fluids,
                                  const float3 gravity)
  {

    thrust::transform(thrust::device, fluids->accPtr(),
                      fluids->accPtr() + fluids->size(), fluids->accPtr(),
                      ThrustHelper::Plus<float3>(gravity));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

} // namespace KIRI
