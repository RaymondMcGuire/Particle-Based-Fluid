/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 17:49:11
 * @LastEditTime: 2021-02-15 14:39:24
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\sph\cuda_wcsph_solver.cu
 */

#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
#include <kiri_pbs_cuda/sph/cuda_wcsph_solver.cuh>
#include <kiri_pbs_cuda/sph/cuda_wcsph_solver_gpu.cuh>
#include <kiri_pbs_cuda/sph/cuda_sph_solver_common_gpu.cuh>
namespace KIRI
{

  void CudaWCSphSolver::ComputeDensity(
      CudaSphParticlesPtr &fluids,
      CudaBoundaryParticlesPtr &boundaries,
      const float rho0,
      const CudaArray<uint> &cellStart,
      const CudaArray<uint> &boundaryCellStart,
      const float3 lowestPoint,
      const float kernelSize,
      const int3 gridSize)
  {
    if (bCubicKernel)
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
          CubicKernel(kernelSize));
    else
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

  void CudaWCSphSolver::ComputeNablaTerm(
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

    ComputePressureByTait_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->GetDensityPtr(),
        fluids->GetPressurePtr(),
        fluids->Size(),
        rho0,
        stiff,
        mNegativeScale);
    if (bCubicKernel)
      ComputeNablaTermConstrain_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
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
          CubicKernelGrad(kernelSize));
    else
      ComputeNablaTermConstrain_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
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

  void CudaWCSphSolver::ComputeViscosityTerm(
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
    if (bCubicKernel)
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
          CubicKernelGrad(kernelSize),
          ViscosityKernelLaplacian(kernelSize));
    else
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
          SpikyKernelLaplacian(kernelSize));
    KIRI_CUKERNAL();
  }

  void CudaWCSphSolver::ComputeArtificialViscosityTerm(
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
    if (bCubicKernel)
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
          CubicKernelGrad(kernelSize));
    else
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

} // namespace KIRI
