/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-25 12:33:06
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-07-26 12:29:31
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_dfsph_particles.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_DFSPH_PARTICLES_CUH_
#define _CUDA_DFSPH_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_sph_particles.cuh>

namespace KIRI
{
  class CudaDFSphParticles final : public CudaSphParticles
  {
  public:
    explicit CudaDFSphParticles::CudaDFSphParticles(const uint numOfMaxParticles)
        : CudaSphParticles(numOfMaxParticles), mAlpha(numOfMaxParticles), mStiff(numOfMaxParticles), mWarmStiff(numOfMaxParticles),
          mVelMag(numOfMaxParticles), mDensityAdv(numOfMaxParticles),
          mDensityError(numOfMaxParticles) {}

    explicit CudaDFSphParticles::CudaDFSphParticles(const Vec_Float3 &p,
                                                    const Vec_Float &mass,
                                                    const Vec_Float &rad,
                                                    const Vec_Float3 &col)
        : CudaSphParticles(p, mass, rad, col), mAlpha(p.size()), mStiff(p.size()), mWarmStiff(p.size()),
          mVelMag(p.size()), mDensityAdv(p.size()), mDensityError(p.size()) {}

    explicit CudaDFSphParticles::CudaDFSphParticles(
        const size_t numOfMaxParticles, const Vec_Float3 &p,
        const Vec_Float &mass, const Vec_Float &rad, const Vec_Float3 &col)
        : CudaSphParticles(numOfMaxParticles, p, mass, rad, col),
          mAlpha(numOfMaxParticles), mStiff(numOfMaxParticles), mWarmStiff(numOfMaxParticles), mVelMag(numOfMaxParticles),
          mDensityAdv(numOfMaxParticles), mDensityError(numOfMaxParticles) {}

    CudaDFSphParticles(const CudaDFSphParticles &) = delete;
    CudaDFSphParticles &operator=(const CudaDFSphParticles &) = delete;

    inline float *alphaPtr() const { return mAlpha.data(); }
    inline float *stiffPtr() const { return mStiff.data(); }
    inline float *warmStiffPtr() const { return mWarmStiff.data(); }
    inline float *velMagPtr() const { return mVelMag.data(); }
    inline float *densityAdvPtr() const { return mDensityAdv.data(); }
    inline float *densityErrorPtr() const { return mDensityError.data(); }

    virtual ~CudaDFSphParticles() noexcept {}

    void velAdvect(const float dt);
    virtual void advect(const float dt) override;

  protected:
    CudaArray<float> mAlpha;
    CudaArray<float> mStiff;
    CudaArray<float> mWarmStiff;
    CudaArray<float> mVelMag;
    CudaArray<float> mDensityAdv;
    CudaArray<float> mDensityError;
  };

  typedef SharedPtr<CudaDFSphParticles> CudaDFSphParticlesPtr;
} // namespace KIRI

#endif