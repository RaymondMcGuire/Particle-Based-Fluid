/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-05-24 09:27:09
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-13 18:00:42
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_pbf_particles.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_PBF_PARTICLES_CUH_
#define _CUDA_PBF_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_sph_particles.cuh>

namespace KIRI
{
  class CudaPBFParticles : public CudaSphParticles
  {
  public:
    explicit CudaPBFParticles::CudaPBFParticles(const uint numOfMaxParticles)
        : CudaSphParticles(numOfMaxParticles),
          mLambda(numOfMaxParticles),
          mDeltaPos(numOfMaxParticles),
          mLastPos(numOfMaxParticles),
          mOldPos(numOfMaxParticles),
          mOmega(numOfMaxParticles),
          mNormOmega(numOfMaxParticles),
          mDensityError(1) {}

    explicit CudaPBFParticles::CudaPBFParticles(
        const Vec_Float3 &pos,
        const Vec_Float &mass,
        const Vec_Float &radius,
        const Vec_Float3 &color)
        : CudaSphParticles(pos, mass, radius, color),
          mLambda(pos.size()),
          mDeltaPos(pos.size()),
          mLastPos(pos.size()),
          mOldPos(pos.size()),
          mOmega(pos.size()),
          mNormOmega(pos.size()),
          mDensityError(1)
    {
      KIRI_CUCALL(cudaMemcpy(mOldPos.data(), &pos[0], sizeof(float3) * pos.size(),
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mLastPos.data(), &pos[0],
                             sizeof(float3) * pos.size(),
                             cudaMemcpyHostToDevice));
    }

    explicit CudaPBFParticles::CudaPBFParticles(
        const size_t numOfMaxParticles,
        const Vec_Float3 &pos,
        const Vec_Float &mass,
        const Vec_Float &radius,
        const Vec_Float3 &color)
        : CudaSphParticles(numOfMaxParticles, pos, mass, radius, color),
          mLambda(numOfMaxParticles),
          mDeltaPos(numOfMaxParticles),
          mLastPos(numOfMaxParticles),
          mOldPos(numOfMaxParticles),
          mOmega(numOfMaxParticles),
          mNormOmega(numOfMaxParticles),
          mDensityError(1)
    {
      KIRI_CUCALL(cudaMemcpy(mOldPos.data(), &pos[0], sizeof(float3) * pos.size(),
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mLastPos.data(), &pos[0],
                             sizeof(float3) * pos.size(),
                             cudaMemcpyHostToDevice));
    }

    CudaPBFParticles(const CudaPBFParticles &) = delete;
    CudaPBFParticles &operator=(const CudaPBFParticles &) = delete;

    inline float *lambdaPtr() const { return mLambda.data(); }
    inline float3 *deltaPosPtr() const { return mDeltaPos.data(); }
    inline float3 *lastPosPtr() const { return mLastPos.data(); }
    inline float3 *oldPosPtr() const { return mOldPos.data(); }
    inline float *densityErrorPtr() const { return mDensityError.data(); }

    inline float3 *omegaPtr() const { return mOmega.data(); }
    inline float3 *normOmegaPtr() const { return mNormOmega.data(); }

    inline float densityErrorSum()
    {
      std::vector<float> density_error_host(1);
      mDensityError.copyToVec(&density_error_host);
      return density_error_host[0];
    }

    virtual ~CudaPBFParticles() noexcept {}

    virtual void advect(const float dt) override;

    virtual void correctPos();
    virtual void updateVelFirstOrder(const float dt);
    virtual void updateVelSecondOrder(const float dt);

  protected:
    CudaArray<float> mDensityError;
    CudaArray<float> mLambda;
    CudaArray<float3> mDeltaPos;
    CudaArray<float3> mLastPos;
    CudaArray<float3> mOldPos;

    CudaArray<float3> mOmega;
    CudaArray<float3> mNormOmega;
  };

  typedef SharedPtr<CudaPBFParticles> CudaPBFParticlesPtr;
} // namespace KIRI

#endif