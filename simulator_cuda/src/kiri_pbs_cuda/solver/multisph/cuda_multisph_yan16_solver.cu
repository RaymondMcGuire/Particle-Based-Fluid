/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 17:49:11
 * @LastEditTime: 2021-04-23 02:14:32
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\multisph\cuda_multisph_yan16_solver.cu
 */

#include <kiri_pbs_cuda/solver/cuda_solver_common_gpu.cuh>
#include <kiri_pbs_cuda/solver/multisph/cuda_multisph_yan16_solver.cuh>
#include <kiri_pbs_cuda/solver/multisph/cuda_multisph_yan16_solver_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>

namespace KIRI
{
  void CudaMultiSphYan16Solver::computeMixDensity(
      CudaMultiSphYan16ParticlesPtr &particles,
      CudaBoundaryParticlesPtr &boundaries, const size_t phaseNum,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize)
  {
    ComputeMultiSphYanMixDensityAndViscosity_CUDA<<<mCudaGridSize,
                                                    KIRI_CUBLOCKSIZE>>>(
        particles->GetYan16PhasePtr(), particles->restRho0Ptr(),
        particles->restVisc0Ptr(), particles->size(), phaseNum);

    ComputeMultiSphYanDensity_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->avgDensityPtr(), particles->posPtr(),
        particles->mixMassPtr(), particles->GetYan16PhasePtr(),
        particles->size(), cellStart.data(), boundaries->posPtr(),
        boundaries->volumePtr(), boundaryCellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize), Poly6Kernel(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphYan16Solver::computeMixPressure(
      CudaMultiSphYan16ParticlesPtr &particles, const size_t phaseNum,
      const bool miscible, const float stiff)
  {

    ComputeMultiSphYanPressure_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetYan16PhasePtr(), particles->avgDensityPtr(),
        particles->size(), phaseNum, miscible, stiff, 1.f);

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphYan16Solver::computeGradientTerm(
      CudaMultiSphYan16ParticlesPtr &particles, const size_t phaseNum,
      const bool miscible, const CudaArray<size_t> &cellStart,
      const float3 lowestPoint, const float kernelRadius, const int3 gridSize)
  {

    ComputeMultiSphYanGradientTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetYan16PhasePtr(), particles->posPtr(),
        particles->mixMassPtr(), particles->avgDensityPtr(),
        particles->size(), phaseNum, miscible, cellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphYan16Solver::computeDriftVelocities(
      CudaMultiSphYan16ParticlesPtr &particles, const size_t phaseNum,
      const bool miscible, const float tou, const float sigma,
      const float3 gravity)
  {

    ComputeMultiSphYanDriftVelocities_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetYan16PhasePtr(), particles->accPtr(),
        particles->restRho0Ptr(), particles->size(), phaseNum, miscible, tou,
        sigma, gravity);

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphYan16Solver::computeDeltaVolumeFraction(
      CudaMultiSphYan16ParticlesPtr &particles, const size_t phaseNum,
      const bool miscible, const CudaArray<size_t> &cellStart,
      const float3 lowestPoint, const float kernelRadius, const int3 gridSize)
  {

    ComputeMultiSphYanDeltaVolumeFraction_CUDA<<<mCudaGridSize,
                                                 KIRI_CUBLOCKSIZE>>>(
        particles->GetYan16PhasePtr(), particles->posPtr(),
        particles->mixMassPtr(), particles->velPtr(),
        particles->avgDensityPtr(), particles->size(), phaseNum, miscible,
        cellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphYan16Solver::correctVolumeFraction(
      CudaMultiSphYan16ParticlesPtr &particles, const size_t phaseNum,
      const bool miscible, const float dt)
  {

    CorrectMultiSphYanVolumeFraction_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetYan16PhasePtr(), particles->size(), phaseNum, miscible, dt);

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphYan16Solver::ComputeDeviatoricStressRateTensor(
      CudaMultiSphYan16ParticlesPtr &particles, const float G,
      const size_t phaseNum, const CudaArray<size_t> &cellStart,
      const float3 lowestPoint, const float kernelRadius, const int3 gridSize)
  {
    ComputeMultiSphYanSolidDeviatoricStressRateTensor_CUDA<<<mCudaGridSize,
                                                             KIRI_CUBLOCKSIZE>>>(
        particles->GetYan16PhasePtr(), particles->posPtr(),
        particles->mixMassPtr(), particles->velPtr(),
        particles->avgDensityPtr(), G, particles->size(), phaseNum,
        cellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphYan16Solver::CorrectDeviatoricStressTensor(
      CudaMultiSphYan16ParticlesPtr &particles, const size_t phaseNum,
      const float Y)
  {
    CorrectMultiSphYanSolidDeviatoricStressTensor_CUDA<<<mCudaGridSize,
                                                         KIRI_CUBLOCKSIZE>>>(
        particles->GetYan16PhasePtr(), particles->size(), phaseNum, Y);

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphYan16Solver::computeMultiSphAcc(
      CudaMultiSphYan16ParticlesPtr &particles,
      CudaBoundaryParticlesPtr &boundaries, const size_t phaseNum,
      const float3 gravity, const float soundSpeed, const float bnu,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize)
  {

    ComputeMultiSphYanAcc_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->accPtr(), particles->posPtr(), particles->mixMassPtr(),
        particles->velPtr(), particles->avgDensityPtr(),
        particles->GetYan16PhasePtr(), particles->restRho0Ptr(), gravity,
        kernelRadius, soundSpeed, bnu, particles->size(), phaseNum,
        cellStart.data(), boundaries->posPtr(), boundaries->volumePtr(),
        boundaryCellStart.data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize), SpikyKernelGrad(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphYan16Solver::computeNextTimeStepData(
      CudaMultiSphYan16ParticlesPtr &particles, const size_t phaseNum)
  {

    ComputeMultiSphYanNextTimeStepData_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->mixMassPtr(), particles->velPtr(), particles->colorPtr(),
        particles->GetYan16PhasePtr(), particles->size(), phaseNum,
        particles->restRho0Ptr(), particles->restMass0Ptr(),
        particles->restColor0Ptr());

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiSphYan16Solver::advect(CudaMultiSphYan16ParticlesPtr &particles,
                                       const float dt, const float3 lowestPoint,
                                       const float3 highestPoint,
                                       const float radius)
  {
    size_t num = particles->size();
    particles->advect(dt);
    _WorldBoundaryConstrain_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->posPtr(), particles->velPtr(), num, lowestPoint, highestPoint,
        radius);

    thrust::fill(thrust::device, particles->avgDensityPtr(),
                 particles->avgDensityPtr() + num, 0.f);
    // thrust::fill(thrust::device, particles->accPtr(), particles->accPtr()
    // + num, make_float3(0.f));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

} // namespace KIRI
