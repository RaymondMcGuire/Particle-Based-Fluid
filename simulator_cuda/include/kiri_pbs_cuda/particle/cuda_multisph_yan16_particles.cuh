/*
 * @Author: Xu.WANG
 * @Date: 2021-02-04 12:36:10
 * @LastEditTime: 2021-04-23 03:09:19
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_multisph_yan16_particles.cuh
 */

#ifndef _CUDA_MULTISPH_YAN16_PARTICLES_CUH_
#define _CUDA_MULTISPH_YAN16_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/data/cuda_multisph_params.h>
#include <kiri_pbs_cuda/particle/cuda_particles.cuh>

namespace KIRI
{
  class CudaMultiSphYan16Particles : public CudaParticles
  {
  public:
    explicit CudaMultiSphYan16Particles::CudaMultiSphYan16Particles(
        const size_t numOfMaxParticles)
        : CudaParticles(numOfMaxParticles), mPhaseLabel(numOfMaxParticles),
          mPhaseType(numOfMaxParticles), mVel(numOfMaxParticles),
          mAcc(numOfMaxParticles), mCol(numOfMaxParticles),
          mAvgDensity(numOfMaxParticles), mMixMass(numOfMaxParticles),
          mYan16PhaseData(numOfMaxParticles),
          mRestPhaseType0(MULTISPH_MAX_PHASE_NUM),
          mRestRho0(MULTISPH_MAX_PHASE_NUM), mRestMass0(MULTISPH_MAX_PHASE_NUM),
          mRestVisc0(MULTISPH_MAX_PHASE_NUM),
          mRestColor0(MULTISPH_MAX_PHASE_NUM) {}

    explicit CudaMultiSphYan16Particles::CudaMultiSphYan16Particles(
        const Vec_Float3 &p, const Vec_Float3 &col, const Vec_Float &mass,
        const Vec_SizeT &phaseLabel, const Vec_SizeT &phaseType,
        const size_t phase0[MULTISPH_MAX_PHASE_NUM],
        const float rho0[MULTISPH_MAX_PHASE_NUM],
        const float mass0[MULTISPH_MAX_PHASE_NUM],
        const float visc0[MULTISPH_MAX_PHASE_NUM],
        const float3 color0[MULTISPH_MAX_PHASE_NUM])
        : CudaParticles(p), mPhaseLabel(p.size()), mPhaseType(p.size()),
          mVel(p.size()), mAcc(p.size()), mCol(p.size()), mAvgDensity(p.size()),
          mMixMass(p.size()), mYan16PhaseData(p.size()),
          mRestPhaseType0(MULTISPH_MAX_PHASE_NUM),
          mRestRho0(MULTISPH_MAX_PHASE_NUM), mRestMass0(MULTISPH_MAX_PHASE_NUM),
          mRestVisc0(MULTISPH_MAX_PHASE_NUM),
          mRestColor0(MULTISPH_MAX_PHASE_NUM)
    {
      KIRI_CUCALL(cudaMemcpy(mRestPhaseType0.data(), &phase0[0],
                             sizeof(size_t) * MULTISPH_MAX_PHASE_NUM,
                             cudaMemcpyHostToDevice));
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
      KIRI_CUCALL(cudaMemcpy(mPhaseType.data(), &phaseType[0],
                             sizeof(size_t) * phaseType.size(),
                             cudaMemcpyHostToDevice));
    }

    explicit CudaMultiSphYan16Particles::CudaMultiSphYan16Particles(
        const size_t numOfMaxParticles, const Vec_Float3 &p,
        const Vec_Float3 &col, const Vec_Float &mass, const Vec_SizeT &phaseLabel,
        const Vec_SizeT &phaseType, const size_t phase0[MULTISPH_MAX_PHASE_NUM],
        const float rho0[MULTISPH_MAX_PHASE_NUM],
        const float mass0[MULTISPH_MAX_PHASE_NUM],
        const float visc0[MULTISPH_MAX_PHASE_NUM],
        const float3 color0[MULTISPH_MAX_PHASE_NUM])
        : CudaParticles(numOfMaxParticles, p), mPhaseLabel(numOfMaxParticles),
          mPhaseType(numOfMaxParticles), mVel(numOfMaxParticles),
          mAcc(numOfMaxParticles), mCol(numOfMaxParticles),
          mAvgDensity(numOfMaxParticles), mMixMass(numOfMaxParticles),
          mYan16PhaseData(numOfMaxParticles),
          mRestPhaseType0(MULTISPH_MAX_PHASE_NUM),
          mRestRho0(MULTISPH_MAX_PHASE_NUM), mRestMass0(MULTISPH_MAX_PHASE_NUM),
          mRestVisc0(MULTISPH_MAX_PHASE_NUM),
          mRestColor0(MULTISPH_MAX_PHASE_NUM)
    {
      KIRI_CUCALL(cudaMemcpy(mRestPhaseType0.data(), &phase0[0],
                             sizeof(size_t) * MULTISPH_MAX_PHASE_NUM,
                             cudaMemcpyHostToDevice));
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
      KIRI_CUCALL(cudaMemcpy(mPhaseType.data(), &phaseType[0],
                             sizeof(size_t) * phaseType.size(),
                             cudaMemcpyHostToDevice));
    }

    CudaMultiSphYan16Particles(const CudaMultiSphYan16Particles &) = delete;
    CudaMultiSphYan16Particles &
    operator=(const CudaMultiSphYan16Particles &) = delete;

    size_t *phaseLabelPtr() const { return mPhaseLabel.data(); }
    size_t *GetPhaseTypePtr() const { return mPhaseType.data(); }

    float3 *velPtr() const { return mVel.data(); }
    float3 *accPtr() const { return mAcc.data(); }
    float3 *colorPtr() const { return mCol.data(); }

    float *avgDensityPtr() const { return mAvgDensity.data(); }
    float *mixMassPtr() const { return mMixMass.data(); }

    size_t *GetRestPhaseType0Ptr() const { return mRestPhaseType0.data(); }
    float *restRho0Ptr() const { return mRestRho0.data(); }
    float *restMass0Ptr() const { return mRestMass0.data(); }
    float *restVisc0Ptr() const { return mRestVisc0.data(); }
    float3 *restColor0Ptr() const { return mRestColor0.data(); }

    Yan16PhaseData *GetYan16PhasePtr() const { return mYan16PhaseData.data(); }

    virtual ~CudaMultiSphYan16Particles() noexcept {}

    virtual void advect(const float dt);

    void AddMultiSphParticles(Vec_Float3 pos, float3 col, float3 vel, float mass);

  protected:
    CudaArray<size_t> mPhaseLabel;
    CudaArray<size_t> mPhaseType;

    CudaArray<float3> mVel;
    CudaArray<float3> mAcc;
    CudaArray<float3> mCol;

    CudaArray<float> mAvgDensity;
    CudaArray<float> mMixMass;

    CudaArray<size_t> mRestPhaseType0;
    CudaArray<float> mRestRho0;
    CudaArray<float> mRestMass0;
    CudaArray<float> mRestVisc0;
    CudaArray<float3> mRestColor0;
    CudaArray<Yan16PhaseData> mYan16PhaseData;
  };

  typedef SharedPtr<CudaMultiSphYan16Particles> CudaMultiSphYan16ParticlesPtr;
} // namespace KIRI

#endif