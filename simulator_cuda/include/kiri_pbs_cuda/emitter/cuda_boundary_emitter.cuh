/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 14:46:49
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-07-08 16:55:06
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\emitter\cuda_boundary_emitter.cuh
 * @Description: 
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved. 
 */

#ifndef _CUDA_BOUNDARY_EMITTER_CUH_
#define _CUDA_BOUNDARY_EMITTER_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {
struct BoundaryData {
  Vec_Float3 pos;
  Vec_SizeT label;
};

class CudaBoundaryEmitter {
public:
  explicit CudaBoundaryEmitter(bool enable = true) : bEnable(enable) {}

  CudaBoundaryEmitter(const CudaBoundaryEmitter &) = delete;
  CudaBoundaryEmitter &operator=(const CudaBoundaryEmitter &) = delete;
  virtual ~CudaBoundaryEmitter() noexcept {}

  void buildWorldBoundary(BoundaryData &data, const float3 &lowest,
                          const float3 &highest, const float particleRadius);
  void buildBoundaryShapeVolume(BoundaryData &data, Vector<float4> shape);

private:
  bool bEnable;
};

typedef SharedPtr<CudaBoundaryEmitter> CudaBoundaryEmitterPtr;
} // namespace KIRI

#endif