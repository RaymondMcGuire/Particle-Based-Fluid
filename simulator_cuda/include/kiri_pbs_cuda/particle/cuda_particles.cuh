/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-24 14:09:08
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-09-29 11:48:46
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_particles.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_PARTICLES_CUH_
#define _CUDA_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/data/cuda_array.cuh>

namespace KIRI {
class CudaParticles {
public:
  explicit CudaParticles(const size_t numOfMaxParticles)
      : mPos(numOfMaxParticles), mParticle2Cell(numOfMaxParticles),
        mNumOfParticles(numOfMaxParticles),
        mNumOfMaxParticles(numOfMaxParticles) {}

  explicit CudaParticles(const Vec_Float3 &p)
      : mPos(p.size()), mParticle2Cell(p.size()), mNumOfParticles(p.size()),
        mNumOfMaxParticles(p.size()) {
    if (!p.empty())
      KIRI_CUCALL(cudaMemcpy(mPos.data(), &p[0], sizeof(float3) * p.size(),
                             cudaMemcpyHostToDevice));
  }

  explicit CudaParticles(const size_t numOfMaxParticles, const Vec_Float3 &p)
      : mPos(numOfMaxParticles), mParticle2Cell(numOfMaxParticles),
        mNumOfParticles(p.size()), mNumOfMaxParticles(numOfMaxParticles) {
    if (!p.empty())
      KIRI_CUCALL(cudaMemcpy(mPos.data(), &p[0], sizeof(float3) * p.size(),
                             cudaMemcpyHostToDevice));
  }

  CudaParticles(const CudaParticles &) = delete;
  CudaParticles &operator=(const CudaParticles &) = delete;

  virtual ~CudaParticles() noexcept {}

  inline size_t size() const { return mNumOfParticles; }
  inline size_t maxSize() const { return mNumOfMaxParticles; }
  inline float3 *posPtr() const { return mPos.data(); }
  inline size_t *particle2CellPtr() const { return mParticle2Cell.data(); }

protected:
  size_t mNumOfParticles;
  size_t mNumOfMaxParticles;
  CudaArray<float3> mPos;
  CudaArray<size_t> mParticle2Cell;
};

typedef SharedPtr<CudaParticles> CudaParticlesPtr;
} // namespace KIRI

#endif