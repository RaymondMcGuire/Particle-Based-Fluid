/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2021-03-29 12:45:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-07-08 13:36:19
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\emitter\cuda_emitter.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_pbs_cuda/emitter/cuda_emitter.cuh>
namespace KIRI {

Vec_Float3 CudaEmitter::Emit() {
  KIRI_PBS_ASSERT(bBuild);

  Vec_Float3 emitPoints;
  for (size_t i = 0; i < mSamples.size(); i++) {
    float3 p =
        mEmitPosition + mSamples[i].x * mEmitAxis1 + mSamples[i].y * mEmitAxis2;
    emitPoints.emplace_back(p);
  }
  return emitPoints;
}

void CudaEmitter::buildSquareEmitter(float particleRadius,
                                     float emitterRadius) {
  mSamples.clear();
  mEmitRadius.clear();

  float offset = particleRadius * 2.f;

  for (float i = -emitterRadius; i < emitterRadius; i += offset) {
    for (float j = -emitterRadius; j < emitterRadius; j += offset) {
      mSamples.emplace_back(make_float2(i, j));
      mEmitRadius.emplace_back(particleRadius);
    }
  }

  if (!mSamples.empty())
    bBuild = true;
}
void CudaEmitter::buildCircleEmitter(float particleRadius,
                                     float emitterRadius) {
  mSamples.clear();
  mEmitRadius.clear();

  float offset = particleRadius * 2.f;

  for (float i = -emitterRadius; i < emitterRadius; i += offset) {
    for (float j = -emitterRadius; j < emitterRadius; j += offset) {
      float2 p = make_float2(i, j);
      if (length(p) <= emitterRadius) {
        mSamples.emplace_back(p);
        mEmitRadius.emplace_back(particleRadius);
      }
    }
  }

  if (!mSamples.empty())
    bBuild = true;
}
void CudaEmitter::buildRectangleEmitter(float particleRadius,
                                        float emitterWidth,
                                        float emitterHeight) {
  mSamples.clear();
  mEmitRadius.clear();

  float offset = particleRadius * 2.f;

  for (float i = -emitterWidth; i < emitterWidth; i += offset) {
    for (float j = -emitterHeight; j < emitterHeight; j += offset) {
      mSamples.emplace_back(make_float2(i, j));
      mEmitRadius.emplace_back(particleRadius);
    }
  }

  if (!mSamples.empty())
    bBuild = true;
}

} // namespace KIRI
