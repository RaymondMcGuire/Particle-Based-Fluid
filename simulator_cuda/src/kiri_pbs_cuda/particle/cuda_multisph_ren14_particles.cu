/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-05-18 21:25:47
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-12 23:25:56
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\particle\cuda_multisph_ren14_particles.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_pbs_cuda/particle/cuda_multisph_ren14_particles.cuh>
namespace KIRI
{
  void CudaMultiSphRen14Particles::advect(const float dt)
  {
    thrust::transform(
        thrust::device, mVel.data(), mVel.data() + size(), mAcc.data(),
        mVel.data(), [dt] __host__ __device__(const float3 &lv, const float3 &a)
        { return lv + dt * a; });

    thrust::transform(
        thrust::device, mPos.data(), mPos.data() + size(), mVel.data(),
        mPos.data(), [dt] __host__ __device__(const float3 &lp, const float3 &v)
        { return lp + dt * v; });

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

} // namespace KIRI
