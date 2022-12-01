/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-05-18 21:25:47
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-06 11:23:49
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\multisph\cuda_multiwcsph_ren14_solver.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_pbs_cuda/solver/multisph/cuda_multiwcsph_ren14_solver.cuh>
#include <kiri_pbs_cuda/solver/multisph/cuda_multiwcsph_ren14_solver_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
#include <thrust/device_ptr.h>

namespace KIRI
{

  void CudaMultiWCSphRen14Solver::computeMixPressure(
      CudaMultiSphRen14ParticlesPtr &particles, const size_t phaseNum,
      const bool miscible, const float stiff)
  {

    ComputeMultiWCSphPressure_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->phaseDataBlock1Ptr(),
        particles->phaseDataBlock2Ptr(),
        particles->avgDensityPtr(),
        particles->size(),
        phaseNum,
        miscible,
        stiff,
        mNegativeScale);

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaMultiWCSphRen14Solver::computeSubTimeStepsByCFL(
      CudaMultiSphRen14ParticlesPtr &particles, const size_t phaseNum,
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
