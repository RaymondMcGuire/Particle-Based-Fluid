/*
 * @Author: Xu.WANG
 * @Date: 2021-02-04 12:36:10
 * @LastEditTime: 2022-04-03 16:31:58
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_iisph_particles.cuh
 */

#ifndef _CUDA_IISPH_PARTICLES_CUH_
#define _CUDA_IISPH_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_sph_particles.cuh>

namespace KIRI
{
  class CudaIISphParticles final : public CudaSphParticles
  {
  public:
    explicit CudaIISphParticles::CudaIISphParticles(const uint numOfMaxParticles)
        : CudaSphParticles(numOfMaxParticles), mAii(numOfMaxParticles),
          mDii(numOfMaxParticles), mDijPj(numOfMaxParticles),
          mDensityAdv(numOfMaxParticles), mLastPressure(numOfMaxParticles),
          mDensityError(numOfMaxParticles), mPressureAcc(numOfMaxParticles) {}

    explicit CudaIISphParticles::CudaIISphParticles(const Vec_Float3 &p,
                                                    const Vec_Float &mass,
                                                    const Vec_Float &rad,
                                                    const Vec_Float3 &col)
        : CudaSphParticles(p, mass, rad, col), mAii(p.size()), mDii(p.size()),
          mDijPj(p.size()), mDensityAdv(p.size()), mLastPressure(p.size()),
          mDensityError(p.size()), mPressureAcc(p.size()) {}

    explicit CudaIISphParticles::CudaIISphParticles(
        const size_t numOfMaxParticles, const Vec_Float3 &p,
        const Vec_Float &mass, const Vec_Float &rad, const Vec_Float3 &col)
        : CudaSphParticles(numOfMaxParticles, p, mass, rad, col),
          mAii(numOfMaxParticles), mDii(numOfMaxParticles),
          mDijPj(numOfMaxParticles), mDensityAdv(numOfMaxParticles),
          mLastPressure(numOfMaxParticles), mDensityError(numOfMaxParticles),
          mPressureAcc(numOfMaxParticles) {}

    CudaIISphParticles(const CudaIISphParticles &) = delete;
    CudaIISphParticles &operator=(const CudaIISphParticles &) = delete;

    inline float *GetAiiPtr() const { return mAii.data(); }
    inline float3 *GetDiiPtr() const { return mDii.data(); }
    inline float3 *GetDijPjPtr() const { return mDijPj.data(); }
    inline float *GetDensityAdvPtr() const { return mDensityAdv.data(); }
    inline float *GetLastPressurePtr() const { return mLastPressure.data(); }
    inline float *GetDensityErrorPtr() const { return mDensityError.data(); }
    inline float3 *GetPressureAccPtr() const { return mPressureAcc.data(); }

    virtual ~CudaIISphParticles() noexcept {}

    void predictVelAdvect(const float dt);
    virtual void advect(const float dt) override;

  protected:
    CudaArray<float> mAii;
    CudaArray<float3> mDii;
    CudaArray<float3> mDijPj;
    CudaArray<float> mDensityAdv;
    CudaArray<float> mLastPressure;
    CudaArray<float> mDensityError;
    CudaArray<float3> mPressureAcc;
  };

  typedef SharedPtr<CudaIISphParticles> CudaIISphParticlesPtr;
} // namespace KIRI

#endif