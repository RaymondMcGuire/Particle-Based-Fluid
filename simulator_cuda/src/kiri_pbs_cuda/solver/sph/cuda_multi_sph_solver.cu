/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-11 11:44:38
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-15 18:14:00
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\sph\cuda_multi_sph_solver.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_pbs_cuda/solver/sph/cuda_multi_sph_solver.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_multi_sph_solver_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>

namespace KIRI
{

  void CudaMultiSphSolver::advectMRSph(CudaSphParticlesPtr &fluids, const float dt,
                                       const float3 lowestPoint,
                                       const float3 highestPoint)
  {
    size_t num = fluids->size();
    fluids->advect(dt);
    _MRBoundaryConstrain_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->posPtr(), fluids->velPtr(), fluids->radiusPtr(), num, lowestPoint,
        highestPoint);

    thrust::fill(thrust::device, fluids->densityPtr(), fluids->densityPtr() + num,
                 0.f);
    thrust::fill(thrust::device, fluids->normalPtr(), fluids->normalPtr() + num,
                 make_float3(0.f));
    thrust::fill(thrust::device, fluids->accPtr(), fluids->accPtr() + num,
                 make_float3(0.f));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphSolver::computeMRDensity(
      CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
      const float rho0, const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize)
  {
    _ComputeMRDensity_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->posPtr(), fluids->massPtr(), fluids->densityPtr(),
        fluids->radiusPtr(), rho0, fluids->size(), cellStart.data(),
        boundaries->posPtr(), boundaries->volumePtr(), boundaryCellStart.data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphSolver::computeMRNablaTerm(
      CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize, const float rho0,
      const float stiff)
  {

    _ComputeMRPressure_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->densityPtr(), fluids->pressurePtr(), fluids->size(), rho0, 3.f);

    _ComputeMRNablaTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->posPtr(), fluids->accPtr(), fluids->massPtr(),
        fluids->densityPtr(), fluids->pressurePtr(), fluids->radiusPtr(), rho0,
        fluids->size(), cellStart.data(), boundaries->posPtr(),
        boundaries->volumePtr(), boundaryCellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphSolver::computeMRViscosityTerm(
      CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float rho0,
      const float visc, const float bnu, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize)
  {
    _ComputeMRViscosityTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->posPtr(), fluids->velPtr(), fluids->accPtr(), fluids->massPtr(),
        fluids->densityPtr(), fluids->radiusPtr(), rho0, visc, bnu,
        fluids->size(), cellStart.data(), boundaries->posPtr(),
        boundaries->volumePtr(), boundaryCellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphSolver::computeMRArtificialViscosityTerm(
      CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float rho0,
      const float nu, const float bnu, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize)
  {
    _ComputeMRArtificialViscosityTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->posPtr(), fluids->velPtr(), fluids->accPtr(), fluids->massPtr(),
        fluids->densityPtr(), fluids->radiusPtr(), rho0, nu, bnu, fluids->size(),
        cellStart.data(), boundaries->posPtr(), boundaries->volumePtr(),
        boundaryCellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

} // namespace KIRI
