/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 14:46:49
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-07-08 16:54:46
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\emitter\cuda_emitter.cuh
 * @Description: 
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved. 
 */
#ifndef _CUDA_EMITTER_CUH_
#define _CUDA_EMITTER_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {

class CudaEmitter {
public:
  explicit CudaEmitter()
      : CudaEmitter(make_float3(0.f), make_float3(1.f, 0.f, 0.f), false)

  {}

  explicit CudaEmitter(float3 emitPosition, float3 emitVelocity, bool enable)
      : mEmitPosition(emitPosition), mEmitVelocity(emitVelocity),
        mEmitAxis1(make_float3(1.f)), bBuild(false), bEnable(enable) {
    mSamples.clear();

    float3 axis = normalize(mEmitVelocity);

    if (abs(axis.x) == 1.f && abs(axis.y) == 0.f && abs(axis.z) == 0.f) {
      mEmitAxis1 = normalize(cross(axis, make_float3(0.f, 1.f, 0.f)));
    } else {
      mEmitAxis1 = normalize(cross(axis, make_float3(1.f, 0.f, 0.f)));
    }

    mEmitAxis2 = normalize(cross(axis, mEmitAxis1));
  }

  CudaEmitter(const CudaEmitter &) = delete;
  CudaEmitter &operator=(const CudaEmitter &) = delete;
  virtual ~CudaEmitter() noexcept {}

  Vec_Float3 Emit();

  inline void setEmitterStatus(const bool enable) { bEnable = enable; }

  void buildSquareEmitter(float particleRadius, float emitterRadius);
  void buildCircleEmitter(float particleRadius, float emitterRadius);
  void buildRectangleEmitter(float particleRadius, float emitterWidth,
                             float emitterHeight);

  inline constexpr bool emitterStatus() const { return bEnable; }
  inline size_t emitterPointsNum() const { return mSamples.size(); }
  inline constexpr float3 emitterPosition() const { return mEmitPosition; }
  inline constexpr float3 emitterVelocity() const { return mEmitVelocity; }
  inline Vec_Float emittRadius() const { return mEmitRadius; }

private:
  bool bEnable, bBuild;
  Vec_Float mEmitRadius;
  Vec_Float2 mSamples;
  float3 mEmitPosition, mEmitVelocity;
  float3 mEmitAxis1, mEmitAxis2;
};

typedef SharedPtr<CudaEmitter> CudaEmitterPtr;
} // namespace KIRI

#endif