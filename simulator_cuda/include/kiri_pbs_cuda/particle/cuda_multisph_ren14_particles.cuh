/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-05-18 21:25:47
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-06 11:21:30
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_multisph_ren14_particles.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_MULTISPH_REN14_PARTICLES_CUH_
#define _CUDA_MULTISPH_REN14_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/data/cuda_multisph_params.h>
#include <kiri_pbs_cuda/particle/cuda_particles.cuh>

namespace KIRI
{
  class CudaMultiSphRen14Particles : public CudaParticles
  {
  public:
    explicit CudaMultiSphRen14Particles::CudaMultiSphRen14Particles(
        const size_t numOfMaxParticles)
        : CudaParticles(numOfMaxParticles), mPhaseLabel(numOfMaxParticles),
          mVel(numOfMaxParticles), mAcc(numOfMaxParticles),
          mCol(numOfMaxParticles), mAvgDensity(numOfMaxParticles),
          mMixMass(numOfMaxParticles), mPhaseData1(numOfMaxParticles), mPhaseData2(numOfMaxParticles),
          mRestRho0(MULTISPH_MAX_PHASE_NUM), mRestMass0(MULTISPH_MAX_PHASE_NUM),
          mRestVisc0(MULTISPH_MAX_PHASE_NUM),
          mRestColor0(MULTISPH_MAX_PHASE_NUM) {}

    explicit CudaMultiSphRen14Particles::CudaMultiSphRen14Particles(
        const Vec_Float3 &p, const Vec_Float3 &col, const Vec_Float &mass,
        const Vec_SizeT &phaseLabel, const float rho0[MULTISPH_MAX_PHASE_NUM],
        const float mass0[MULTISPH_MAX_PHASE_NUM],
        const float visc0[MULTISPH_MAX_PHASE_NUM],
        const float3 color0[MULTISPH_MAX_PHASE_NUM])
        : CudaParticles(p), mPhaseLabel(p.size()), mVel(p.size()), mAcc(p.size()),
          mCol(p.size()), mAvgDensity(p.size()), mMixMass(p.size()),
          mPhaseData1(p.size()), mPhaseData2(p.size()), mRestRho0(MULTISPH_MAX_PHASE_NUM),
          mRestMass0(MULTISPH_MAX_PHASE_NUM), mRestVisc0(MULTISPH_MAX_PHASE_NUM),
          mRestColor0(MULTISPH_MAX_PHASE_NUM)
    {
      KIRI_CUCALL(cudaMemcpy(mRestRho0.data(), &rho0[0],
                             sizeof(float) * MULTISPH_MAX_PHASE_NUM,
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mRestMass0.data(), &mass0[0],
                             sizeof(float) * MULTISPH_MAX_PHASE_NUM,
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mRestVisc0.data(), &visc0[0],
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
    }

    explicit CudaMultiSphRen14Particles::CudaMultiSphRen14Particles(
        const size_t numOfMaxParticles, const Vec_Float3 &p,
        const Vec_Float3 &col, const Vec_Float &mass, const Vec_SizeT &phaseLabel,
        const float rho0[MULTISPH_MAX_PHASE_NUM],
        const float mass0[MULTISPH_MAX_PHASE_NUM],
        const float visc0[MULTISPH_MAX_PHASE_NUM],
        const float3 color0[MULTISPH_MAX_PHASE_NUM])
        : CudaParticles(numOfMaxParticles, p), mPhaseLabel(numOfMaxParticles),
          mVel(numOfMaxParticles), mAcc(numOfMaxParticles),
          mCol(numOfMaxParticles), mAvgDensity(numOfMaxParticles),
          mMixMass(numOfMaxParticles), mPhaseData1(numOfMaxParticles), mPhaseData2(numOfMaxParticles),
          mRestRho0(MULTISPH_MAX_PHASE_NUM), mRestMass0(MULTISPH_MAX_PHASE_NUM),
          mRestVisc0(MULTISPH_MAX_PHASE_NUM),
          mRestColor0(MULTISPH_MAX_PHASE_NUM)
    {
      KIRI_CUCALL(cudaMemcpy(mRestRho0.data(), &rho0[0],
                             sizeof(float) * MULTISPH_MAX_PHASE_NUM,
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mRestMass0.data(), &mass0[0],
                             sizeof(float) * MULTISPH_MAX_PHASE_NUM,
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mRestVisc0.data(), &visc0[0],
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
    }

    CudaMultiSphRen14Particles(const CudaMultiSphRen14Particles &) = delete;
    CudaMultiSphRen14Particles &
    operator=(const CudaMultiSphRen14Particles &) = delete;

    size_t *phaseLabelPtr() const { return mPhaseLabel.data(); }

    float3 *velPtr() const { return mVel.data(); }
    float3 *accPtr() const { return mAcc.data(); }
    float3 *colorPtr() const { return mCol.data(); }

    float *avgDensityPtr() const { return mAvgDensity.data(); }
    float *mixMassPtr() const { return mMixMass.data(); }

    float *restRho0Ptr() const { return mRestRho0.data(); }
    float *restMass0Ptr() const { return mRestMass0.data(); }
    float *restVisc0Ptr() const { return mRestVisc0.data(); }
    float3 *restColor0Ptr() const { return mRestColor0.data(); }

    Ren14PhaseDataBlock1 *phaseDataBlock1Ptr() const { return mPhaseData1.data(); }
    Ren14PhaseDataBlock2 *phaseDataBlock2Ptr() const { return mPhaseData2.data(); }

    virtual ~CudaMultiSphRen14Particles() noexcept {}

    virtual void advect(const float dt);

  protected:
    CudaArray<size_t> mPhaseLabel;

    CudaArray<float3> mVel;
    CudaArray<float3> mAcc;
    CudaArray<float3> mCol;

    CudaArray<float> mAvgDensity;
    CudaArray<float> mMixMass;

    CudaArray<float> mRestRho0;
    CudaArray<float> mRestMass0;
    CudaArray<float> mRestVisc0;
    CudaArray<float3> mRestColor0;
    CudaArray<Ren14PhaseDataBlock1> mPhaseData1;
    CudaArray<Ren14PhaseDataBlock2> mPhaseData2;
  };

  typedef SharedPtr<CudaMultiSphRen14Particles> CudaMultiSphRen14ParticlesPtr;
} // namespace KIRI

#endif