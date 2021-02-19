/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 17:49:11
 * @LastEditTime: 2021-02-15 15:18:49
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\sph\cuda_sph_solver.cu
 */

#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
#include <kiri_pbs_cuda/sph/cuda_sph_solver.cuh>
#include <kiri_pbs_cuda/sph/cuda_sph_solver_gpu.cuh>
#include <kiri_pbs_cuda/sph/cuda_sph_solver_common_gpu.cuh>
namespace KIRI
{
  void CudaSphSolver::ComputeDensity(
      CudaSphParticlesPtr &fluids,
      CudaBoundaryParticlesPtr &boundaries,
      const float rho0,
      const CudaArray<uint> &cellStart,
      const CudaArray<uint> &boundaryCellStart,
      const float3 lowestPoint,
      const float kernelSize,
      const int3 gridSize)
  {
    ComputeDensity_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->GetPosPtr(),
        fluids->GetMassPtr(),
        fluids->GetDensityPtr(),
        rho0,
        fluids->Size(),
        cellStart.Data(),
        boundaries->GetPosPtr(),
        boundaries->GetVolumePtr(),
        boundaryCellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelSize, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        Poly6Kernel(kernelSize));

    KIRI_CUKERNAL();
  }

  void CudaSphSolver::ComputeNablaTerm(
      CudaSphParticlesPtr &fluids,
      CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<uint> &cellStart,
      const CudaArray<uint> &boundaryCellStart,
      const float3 lowestPoint,
      const float kernelSize,
      const int3 gridSize,
      const float rho0,
      const float stiff)
  {

    ComputePressure_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->GetDensityPtr(),
        fluids->GetPressurePtr(),
        fluids->Size(),
        rho0,
        stiff);

    ComputeNablaTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->GetPosPtr(),
        fluids->GetAccPtr(),
        fluids->GetMassPtr(),
        fluids->GetDensityPtr(),
        fluids->GetPressurePtr(),
        rho0,
        fluids->Size(),
        cellStart.Data(),
        boundaries->GetPosPtr(),
        boundaries->GetVolumePtr(),
        boundaryCellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelSize, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        SpikyKernelGrad(kernelSize));
    KIRI_CUKERNAL();
  }

  void CudaSphSolver::ComputeViscosityTerm(
      CudaSphParticlesPtr &fluids,
      CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<uint> &cellStart,
      const CudaArray<uint> &boundaryCellStart,
      const float rho0,
      const float visc,
      const float bnu,
      const float3 lowestPoint,
      const float kernelSize,
      const int3 gridSize)
  {
    ComputeViscosityTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->GetPosPtr(),
        fluids->GetVelPtr(),
        fluids->GetAccPtr(),
        fluids->GetMassPtr(),
        fluids->GetDensityPtr(),
        rho0,
        visc,
        bnu,
        fluids->Size(),
        cellStart.Data(),
        boundaries->GetPosPtr(),
        boundaries->GetVolumePtr(),
        boundaryCellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelSize, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        SpikyKernelGrad(kernelSize),
        ViscosityKernelLaplacian(kernelSize));
    KIRI_CUKERNAL();
  }

  void CudaSphSolver::ComputeArtificialViscosityTerm(
      CudaSphParticlesPtr &fluids,
      CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<uint> &cellStart,
      const CudaArray<uint> &boundaryCellStart,
      const float rho0,
      const float nu,
      const float bnu,
      const float3 lowestPoint,
      const float kernelSize,
      const int3 gridSize)
  {
    ComputeArtificialViscosityTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->GetPosPtr(),
        fluids->GetVelPtr(),
        fluids->GetAccPtr(),
        fluids->GetMassPtr(),
        fluids->GetDensityPtr(),
        rho0,
        nu,
        bnu,
        fluids->Size(),
        cellStart.Data(),
        boundaries->GetPosPtr(),
        boundaries->GetVolumePtr(),
        boundaryCellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelSize, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        SpikyKernelGrad(kernelSize));

    KIRI_CUKERNAL();
  }

  void CudaSphSolver::Advect(
      CudaSphParticlesPtr &fluids,
      const float dt,
      const float3 lowestPoint,
      const float3 highestPoint,
      const float radius)
  {
    uint num = fluids->Size();
    fluids->Advect(dt);
    BoundaryConstrain_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->GetPosPtr(),
        fluids->GetVelPtr(),
        num,
        lowestPoint,
        highestPoint,
        radius);

    thrust::fill(thrust::device, fluids->GetDensityPtr(), fluids->GetDensityPtr() + num, 0.f);
    thrust::fill(thrust::device, fluids->GetAccPtr(), fluids->GetAccPtr() + num, make_float3(0.f));
    KIRI_CUKERNAL();
  }

  void CudaSphSolver::ExtraForces(
      CudaSphParticlesPtr &fluids,
      const float3 gravity)
  {

    thrust::transform(thrust::device,
                      fluids->GetAccPtr(), fluids->GetAccPtr() + fluids->Size(),
                      fluids->GetAccPtr(),
                      ThrustHelper::Plus<float3>(gravity));

    KIRI_CUKERNAL();
  }

} // namespace KIRI
