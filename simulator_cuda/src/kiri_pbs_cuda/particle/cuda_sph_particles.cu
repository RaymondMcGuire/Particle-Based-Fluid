/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2021-03-29 12:45:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-07-08 12:58:46
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\particle\cuda_sph_particles.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_pbs_cuda/particle/cuda_sph_particles.cuh>
namespace KIRI
{
  void CudaSphParticles::appendParticles(Vec_Float3 pos, Vec_Float radius,
                                         float3 col, float3 vel, float mass)
  {
    size_t num = pos.size();

    if (this->size() + num >= this->maxSize())
    {
      printf("Current SPH particles numbers exceed maximum particles. \n");
      return;
    }

    KIRI_CUCALL(cudaMemcpy(this->posPtr() + this->size(), &pos[0],
                           sizeof(float3) * num, cudaMemcpyHostToDevice));
    KIRI_CUCALL(cudaMemcpy(this->radiusPtr() + this->size(), &radius[0],
                           sizeof(float) * num, cudaMemcpyHostToDevice));
    thrust::fill(thrust::device, this->colorPtr() + this->size(),
                 this->colorPtr() + this->size() + num, col);
    thrust::fill(thrust::device, this->velPtr() + this->size(),
                 this->velPtr() + this->size() + num, vel);
    thrust::fill(thrust::device, this->massPtr() + this->size(),
                 this->massPtr() + this->size() + num, mass);

    mNumOfParticles += num;

    // printf("mNumOfParticles=%zd \n", mNumOfParticles);
  }

  void CudaSphParticles::advect(const float dt)
  {

    // printf("advect=%zd \n", mPos.Length());
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
