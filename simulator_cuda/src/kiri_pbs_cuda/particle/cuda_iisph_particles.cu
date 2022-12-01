/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 14:33:32
 * @LastEditTime: 2021-07-18 12:12:18
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\particle\cuda_iisph_particles.cu
 */

#include <kiri_pbs_cuda/particle/cuda_iisph_particles.cuh>

#include <thrust/device_ptr.h>

namespace KIRI
{

  void CudaIISphParticles::predictVelAdvect(const float dt)
  {
    thrust::transform(
        thrust::device, mVel.data(), mVel.data() + size(), mAcc.data(),
        mVel.data(), [dt] __host__ __device__(const float3 &lv, const float3 &a)
        { return lv + dt * a; });

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaIISphParticles::advect(const float dt)
  {
    // add pressure acceleration
    thrust::transform(
        thrust::device, mVel.data(), mVel.data() + size(), mPressureAcc.data(),
        mVel.data(),
        [dt] __host__ __device__(const float3 &lv, const float3 &pa)
        {
          return lv + dt * pa;
        });

    thrust::transform(
        thrust::device, mPos.data(), mPos.data() + size(), mVel.data(),
        mPos.data(), [dt] __host__ __device__(const float3 &lp, const float3 &v)
        { return lp + dt * v; });

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

} // namespace KIRI
