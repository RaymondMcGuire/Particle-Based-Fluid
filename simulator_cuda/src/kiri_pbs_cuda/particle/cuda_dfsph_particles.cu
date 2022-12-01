/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-25 12:33:07
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-07-26 15:57:44
 * @FilePath:
 * \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\particle\cuda_dfsph_particles.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_pbs_cuda/particle/cuda_dfsph_particles.cuh>

#include <thrust/device_ptr.h>

namespace KIRI
{

  void CudaDFSphParticles::velAdvect(const float dt)
  {
    thrust::transform(
        thrust::device, mVel.data(), mVel.data() + size(), mAcc.data(),
        mVel.data(), [dt] __host__ __device__(const float3 &lv, const float3 &a)
        { return lv + dt * a; });

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaDFSphParticles::advect(const float dt)
  {

    thrust::transform(
        thrust::device, mPos.data(), mPos.data() + size(), mVel.data(),
        mPos.data(), [dt] __host__ __device__(const float3 &lp, const float3 &v)
        { return lp + dt * v; });

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

} // namespace KIRI
