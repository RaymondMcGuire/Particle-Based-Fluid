/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-12 23:25:56
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-13 18:05:15
 * @FilePath:
 * \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\particle\cuda_multisph_yang15_particles.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_pbs_cuda/particle/cuda_multisph_yang15_particles.cuh>

#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
#include <thrust/device_ptr.h>

namespace KIRI {
struct VelocityUpdateFirstOrder {
  float mDt;
  __host__ __device__ VelocityUpdateFirstOrder(const float dt) : mDt(dt) {}

  __host__ __device__ float3
  operator()(const ThrustHelper::TUPLE_FLOAT3X2 &data) const {
    float3 pos = data.__get_impl<0>();
    float3 old_pos = data.__get_impl<1>();

    return (1.f / mDt) * (pos - old_pos);
  }
};

struct VelocityUpdateSecondOrder {
  float mDt;
  __host__ __device__ VelocityUpdateSecondOrder(const float dt) : mDt(dt) {}

  __host__ __device__ float3
  operator()(const ThrustHelper::TUPLE_FLOAT3X3 &data) const {
    float3 pos = data.__get_impl<0>();
    float3 old_pos = data.__get_impl<1>();
    float3 last_pos = data.__get_impl<2>();

    return (1.f / mDt) * (1.5f * pos - 2.f * old_pos + 0.5f * last_pos);
  }
};

void CudaMultiSphYang15Particles::advect(const float dt) {

  thrust::transform(thrust::device, mOldPos.data(), mOldPos.data() + size(),
                    mLastPos.data(),
                    [] __host__ __device__(const float3 &old) { return old; });

  thrust::transform(thrust::device, mPos.data(), mPos.data() + size(),
                    mOldPos.data(),
                    [] __host__ __device__(const float3 &pos) { return pos; });

  thrust::transform(
      thrust::device, mVel.data(), mVel.data() + size(), mAcc.data(),
      mVel.data(), [dt] __host__ __device__(const float3 &lv, const float3 &a) {
        return lv + dt * a;
      });

  thrust::transform(
      thrust::device, mPos.data(), mPos.data() + size(), mVel.data(),
      mPos.data(), [dt] __host__ __device__(const float3 &lp, const float3 &v) {
        return lp + dt * v;
      });

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaMultiSphYang15Particles::correctPos() {
  thrust::transform(
      thrust::device, mPos.data(), mPos.data() + size(), mDeltaPos.data(),
      mPos.data(),
      [] __host__ __device__(const float3 &lp, const float3 &delta) {
        return lp + delta;
      });

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaMultiSphYang15Particles::updateVelFirstOrder(const float dt) {
  auto data = thrust::make_zip_iterator(
      thrust::make_tuple(mPos.data(), mOldPos.data()));

  thrust::transform(thrust::device, data, data + size(), mVel.data(),
                    VelocityUpdateFirstOrder(dt));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaMultiSphYang15Particles::updateVelSecondOrder(const float dt) {
  auto data = thrust::make_zip_iterator(
      thrust::make_tuple(mPos.data(), mOldPos.data(), mLastPos.data()));

  thrust::transform(thrust::device, data, data + size(), mVel.data(),
                    VelocityUpdateSecondOrder(dt));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

} // namespace KIRI
