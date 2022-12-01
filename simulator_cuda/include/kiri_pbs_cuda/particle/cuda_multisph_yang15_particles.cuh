/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-12 12:56:39
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-12 12:58:27
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_multisph_yang15_particles.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_MULTISPH_YANG15_PARTICLES_CUH_
#define _CUDA_MULTISPH_YANG15_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/data/cuda_multisph_params.h>
#include <kiri_pbs_cuda/particle/cuda_particles.cuh>

namespace KIRI
{
  class CudaMultiSphYang15Particles : public CudaParticles
  {
  public:
    explicit CudaMultiSphYang15Particles::CudaMultiSphYang15Particles(
        const size_t numOfMaxParticles)
        : CudaParticles(numOfMaxParticles),
          mPhaseLabel(numOfMaxParticles),
          mVel(numOfMaxParticles),
          mAcc(numOfMaxParticles),
          mCol(numOfMaxParticles),
          mAvgDensity(numOfMaxParticles),
          mMixMass(numOfMaxParticles),
          mPhaseData1(numOfMaxParticles),
          mPhaseData2(numOfMaxParticles),
          mLambda(numOfMaxParticles),
          mDeltaPos(numOfMaxParticles),
          mLastPos(numOfMaxParticles),
          mOldPos(numOfMaxParticles),
          mDensityError(1),
          mRestRho0(MULTISPH_MAX_PHASE_NUM),
          mRestMass0(MULTISPH_MAX_PHASE_NUM),
          mRestColor0(MULTISPH_MAX_PHASE_NUM)
    {
    }

    explicit CudaMultiSphYang15Particles::CudaMultiSphYang15Particles(
        const Vec_Float3 &pos,
        const Vec_Float3 &col,
        const Vec_Float &mass,
        const Vec_SizeT &phaseLabel,
        const float rho0[MULTISPH_MAX_PHASE_NUM],
        const float mass0[MULTISPH_MAX_PHASE_NUM],
        const float3 color0[MULTISPH_MAX_PHASE_NUM])
        : CudaParticles(pos),
          mPhaseLabel(pos.size()),
          mVel(pos.size()),
          mAcc(pos.size()),
          mCol(pos.size()),
          mAvgDensity(pos.size()),
          mMixMass(pos.size()),
          mPhaseData1(pos.size()),
          mPhaseData2(pos.size()),
          mLambda(pos.size()),
          mDeltaPos(pos.size()),
          mLastPos(pos.size()),
          mOldPos(pos.size()),
          mDensityError(1),
          mRestRho0(MULTISPH_MAX_PHASE_NUM),
          mRestMass0(MULTISPH_MAX_PHASE_NUM),
          mRestColor0(MULTISPH_MAX_PHASE_NUM)
    {
      KIRI_CUCALL(cudaMemcpy(mRestRho0.data(), &rho0[0],
                             sizeof(float) * MULTISPH_MAX_PHASE_NUM,
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mRestMass0.data(), &mass0[0],
                             sizeof(float) * MULTISPH_MAX_PHASE_NUM,
                             cudaMemcpyHostToDevice));

      KIRI_CUCALL(cudaMemcpy(mRestColor0.data(), &color0[0],
                             sizeof(float3) * MULTISPH_MAX_PHASE_NUM,
                             cudaMemcpyHostToDevice));

      KIRI_CUCALL(cudaMemcpy(mCol.data(), &col[0], sizeof(float3) * col.size(),
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mMixMass.data(), &mass[0],
                             sizeof(float) * mass.size(),
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mPhaseLabel.data(), &phaseLabel[0],
                             sizeof(size_t) * phaseLabel.size(),
                             cudaMemcpyHostToDevice));

      KIRI_CUCALL(cudaMemcpy(mOldPos.data(), &pos[0], sizeof(float3) * pos.size(),
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mLastPos.data(), &pos[0],
                             sizeof(float3) * pos.size(),
                             cudaMemcpyHostToDevice));
    }

    explicit CudaMultiSphYang15Particles::CudaMultiSphYang15Particles(
        const size_t numOfMaxParticles,
        const Vec_Float3 &pos,
        const Vec_Float3 &col,
        const Vec_Float &mass,
        const Vec_SizeT &phaseLabel,
        const float rho0[MULTISPH_MAX_PHASE_NUM],
        const float mass0[MULTISPH_MAX_PHASE_NUM],
        const float3 color0[MULTISPH_MAX_PHASE_NUM])
        : CudaParticles(numOfMaxParticles, pos),
          mPhaseLabel(numOfMaxParticles),
          mVel(numOfMaxParticles),
          mAcc(numOfMaxParticles),
          mCol(numOfMaxParticles),
          mAvgDensity(numOfMaxParticles),
          mMixMass(numOfMaxParticles),
          mPhaseData1(numOfMaxParticles),
          mPhaseData2(numOfMaxParticles),
          mLambda(numOfMaxParticles),
          mDeltaPos(numOfMaxParticles),
          mLastPos(numOfMaxParticles),
          mOldPos(numOfMaxParticles),
          mDensityError(1),
          mRestRho0(MULTISPH_MAX_PHASE_NUM),
          mRestMass0(MULTISPH_MAX_PHASE_NUM),
          mRestColor0(MULTISPH_MAX_PHASE_NUM)
    {
      KIRI_CUCALL(cudaMemcpy(mRestRho0.data(), &rho0[0],
                             sizeof(float) * MULTISPH_MAX_PHASE_NUM,
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mRestMass0.data(), &mass0[0],
                             sizeof(float) * MULTISPH_MAX_PHASE_NUM,
                             cudaMemcpyHostToDevice));

      KIRI_CUCALL(cudaMemcpy(mRestColor0.data(), &color0[0],
                             sizeof(float3) * MULTISPH_MAX_PHASE_NUM,
                             cudaMemcpyHostToDevice));

      KIRI_CUCALL(cudaMemcpy(mCol.data(), &col[0], sizeof(float3) * col.size(),
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mMixMass.data(), &mass[0],
                             sizeof(float) * mass.size(),
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mPhaseLabel.data(), &phaseLabel[0],
                             sizeof(size_t) * phaseLabel.size(),
                             cudaMemcpyHostToDevice));

      KIRI_CUCALL(cudaMemcpy(mOldPos.data(), &pos[0], sizeof(float3) * pos.size(),
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mLastPos.data(), &pos[0],
                             sizeof(float3) * pos.size(),
                             cudaMemcpyHostToDevice));
    }

    CudaMultiSphYang15Particles(const CudaMultiSphYang15Particles &) = delete;
    CudaMultiSphYang15Particles &
    operator=(const CudaMultiSphYang15Particles &) = delete;

    size_t *phaseLabelPtr() const { return mPhaseLabel.data(); }

    float3 *velPtr() const { return mVel.data(); }
    float3 *accPtr() const { return mAcc.data(); }
    float3 *colorPtr() const { return mCol.data(); }

    float *avgDensityPtr() const { return mAvgDensity.data(); }
    float *mixMassPtr() const { return mMixMass.data(); }

    float *restRho0Ptr() const { return mRestRho0.data(); }
    float *restMass0Ptr() const { return mRestMass0.data(); }
    float3 *restColor0Ptr() const { return mRestColor0.data(); }

    Yang15PhaseDataBlock1 *phaseDataBlock1Ptr() const { return mPhaseData1.data(); }
    Yang15PhaseDataBlock2 *phaseDataBlock2Ptr() const { return mPhaseData2.data(); }

    inline float *lambdaPtr() const { return mLambda.data(); }
    inline float3 *deltaPosPtr() const { return mDeltaPos.data(); }
    inline float3 *lastPosPtr() const { return mLastPos.data(); }
    inline float3 *oldPosPtr() const { return mOldPos.data(); }
    inline float *densityErrorPtr() const { return mDensityError.data(); }
    inline float densityErrorSum()
    {
      std::vector<float> density_error_host(1);
      mDensityError.copyToVec(&density_error_host);
      return density_error_host[0];
    }

    virtual ~CudaMultiSphYang15Particles() noexcept {}

    virtual void advect(const float dt);
    virtual void correctPos();
    virtual void updateVelFirstOrder(const float dt);
    virtual void updateVelSecondOrder(const float dt);

  protected:
    CudaArray<size_t> mPhaseLabel;

    CudaArray<float3> mVel;
    CudaArray<float3> mAcc;
    CudaArray<float3> mCol;

    CudaArray<float> mAvgDensity;
    CudaArray<float> mMixMass;

    CudaArray<float> mRestRho0;
    CudaArray<float> mRestMass0;
    CudaArray<float3> mRestColor0;

    CudaArray<Yang15PhaseDataBlock1> mPhaseData1;
    CudaArray<Yang15PhaseDataBlock2> mPhaseData2;

    // pbf
    CudaArray<float> mDensityError;
    CudaArray<float> mLambda;
    CudaArray<float3> mDeltaPos;
    CudaArray<float3> mLastPos;
    CudaArray<float3> mOldPos;
  };

  typedef SharedPtr<CudaMultiSphYang15Particles> CudaMultiSphYang15ParticlesPtr;
} // namespace KIRI

#endif