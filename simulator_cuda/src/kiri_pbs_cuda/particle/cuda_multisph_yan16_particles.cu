/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 14:33:32
 * @LastEditTime: 2021-04-23 00:42:44
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\particle\cuda_multisph_yan16_particles.cu
 */

#include <kiri_pbs_cuda/particle/cuda_multisph_yan16_particles.cuh>
namespace KIRI
{
  void CudaMultiSphYan16Particles::AddMultiSphParticles(Vec_Float3 pos,
                                                        float3 col, float3 vel,
                                                        float mass)
  {
    size_t num = pos.size();

    if (this->size() + num >= this->maxSize())
    {
      printf("Current Multi-SPH Yan16 particles numbers exceed maximum "
             "particles. \n");
      return;
    }

    KIRI_CUCALL(cudaMemcpy(this->posPtr() + this->size(), &pos[0],
                           sizeof(float3) * num, cudaMemcpyHostToDevice));
    thrust::fill(thrust::device, this->colorPtr() + this->size(),
                 this->colorPtr() + this->size() + num, col);
    thrust::fill(thrust::device, this->velPtr() + this->size(),
                 this->velPtr() + this->size() + num, vel);
    thrust::fill(thrust::device, this->mixMassPtr() + this->size(),
                 this->mixMassPtr() + this->size() + num, mass);

    mNumOfParticles += num;
  }

  void CudaMultiSphYan16Particles::advect(const float dt)
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
