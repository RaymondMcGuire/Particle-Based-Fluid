/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 17:49:11
 * @LastEditTime: 2021-04-23 10:30:11
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\multisph\cuda_multiwcsph_yan16_solver.cu
 */

#include <kiri_pbs_cuda/solver/multisph/cuda_multiwcsph_yan16_solver.cuh>
#include <kiri_pbs_cuda/solver/multisph/cuda_multiwcsph_yan16_solver_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
#include <thrust/device_ptr.h>

namespace KIRI
{

  void CudaMultiWCSphYan16Solver::computeMixPressure(
      CudaMultiSphYan16ParticlesPtr &particles, const size_t phaseNum,
      const bool miscible, const float stiff)
  {

    ComputeMultiWCSphYanPressure_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetYan16PhasePtr(), particles->avgDensityPtr(),
        particles->size(), phaseNum, miscible, stiff, mNegativeScale);

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiWCSphYan16Solver::computeSubTimeStepsByCFL(
      CudaMultiSphYan16ParticlesPtr &particles, const size_t phaseNum,
      const float sphMass[MULTISPH_MAX_PHASE_NUM], const float dt,
      const float kernelRadius, float renderInterval)
  {

    auto accArray = thrust::device_pointer_cast(particles->accPtr());
    float3 maxAcc =
        *(thrust::max_element(accArray, accArray + particles->size(),
                              ThrustHelper::CompareLengthCuda<float3>()));

    float mass = 0.f;
    for (size_t i = 0; i < phaseNum; i++)
    {
      mass = std::max(sphMass[i], mass);
    }

    float maxForceMagnitude = length(maxAcc) * mass;
    float timeStepLimitBySpeed =
        mTimeStepLimitBySpeedFactor * kernelRadius / mSpeedOfSound;
    float timeStepLimitByForce =
        mTimeStepLimitByForceFactor *
        std::sqrt(kernelRadius * mass / maxForceMagnitude);
    float desiredTimeStep =
        std::min(mTimeStepLimitScale *
                     std::min(timeStepLimitBySpeed, timeStepLimitByForce),
                 dt);

    mNumOfSubTimeSteps = static_cast<size_t>(renderInterval / desiredTimeStep);

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

} // namespace KIRI
