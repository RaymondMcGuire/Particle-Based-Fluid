/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 14:33:32
 * @LastEditTime: 2021-02-09 12:50:50
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\particle\cuda_sph_particles.cu
 */

#include <kiri_pbs_cuda/particle/cuda_sph_particles.cuh>
namespace KIRI
{

    void CudaSphParticles::Advect(const float dt)
    {
        thrust::transform(thrust::device,
                          mVel.Data(), mVel.Data() + Size(),
                          mAcc.Data(),
                          mVel.Data(),
                          [dt] __host__ __device__(const float3 &lv, const float3 &a) {
                              return lv + dt * a;
                          });

        thrust::transform(thrust::device,
                          mPos.Data(), mPos.Data() + Size(),
                          mVel.Data(),
                          mPos.Data(),
                          [dt] __host__ __device__(const float3 &lp, const float3 &v) {
                              return lp + dt * v;
                          });
    }

} // namespace KIRI
