/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 19:05:14
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-27 11:55:14
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\emitter\cuda_volume_emitter.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_VOLUME_EMITTER_CUH_
#define _CUDA_VOLUME_EMITTER_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{

  struct SphVolumeData
  {
    Vec_Float3 pos;
    Vec_Float mass;
    Vec_Float radius;
    Vec_Float3 col;
    float maxRadius;
    float minRadius;
  };

  struct MultiSphRen14VolumeData
  {
    Vec_Float3 pos;
    Vec_Float3 col;
    Vec_Float mass;
    Vec_SizeT phaseLabel;
  };

  struct MultiSphYan16VolumeData
  {
    Vec_Float3 pos;
    Vec_Float3 col;
    Vec_Float mass;
    Vec_SizeT phaseLabel;
    Vec_SizeT phaseType;
  };

  struct DemVolumeData
  {
    Vec_Float3 pos;
    Vec_Float3 col;
    Vec_Float mass;
  };

  struct DemShapeVolumeData
  {
    float maxRadius;
    float minRadius;
    Vec_Float3 pos;
    Vec_Float3 col;
    Vec_Float mass;
    Vec_Float radius;
  };

  struct SeepageflowVolumeData
  {
    float sandMinRadius;
    Vec_Float3 pos;
    Vec_Float3 col;
    Vec_Float mass;
    Vec_Float radius;
    Vec_SizeT label;
  };

  struct SeepageflowMultiVolumeData
  {
    float sandMinRadius;
    Vec_Float3 pos;
    Vec_Float3 col;
    Vec_Float mass;
    Vec_Float radius;
    Vec_SizeT label;
    Vec_Float2 amcamcp;
    Vec_Float3 cda0asat;
  };

  class CudaVolumeEmitter
  {
  public:
    explicit CudaVolumeEmitter(bool enable = true) : bEnable(enable) {}

    CudaVolumeEmitter(const CudaVolumeEmitter &) = delete;
    CudaVolumeEmitter &operator=(const CudaVolumeEmitter &) = delete;
    virtual ~CudaVolumeEmitter() noexcept {}

    void buildSphVolume(
        SphVolumeData &data,
        float3 lowest,
        int3 vsize,
        float particleRadius,
        float particleMass,
        float3 color);

    void buildSphShapeVolume(
        SphVolumeData &data,
        Vector<float4> shape,
        Vec_Float mass,
        float3 color);

    void buildSphShapeVolume(
        SphVolumeData &data,
        Vector<float4> shape,
        float particleMass,
        float3 color);

    void buildUniDemVolume(DemVolumeData &data, float3 lowest, int3 vsize,
                           float particleRadius, float3 color, float mass,
                           float jitter = 0.001f);
    void buildMRDemVolume(DemShapeVolumeData &data, float3 lowest, int3 vsize,
                          float particleRadius, float3 color, float mass,
                          float jitter = 0.001f);
    void buildDemShapeVolume(DemShapeVolumeData &data, Vector<float4> shape,
                             float3 color, float density);

    void buildSeepageflowBoxVolume(SeepageflowVolumeData &data, float3 lowest,
                                   int3 vsize, float particleRadius, float3 color,
                                   float mass, size_t label,
                                   float jitter = 0.001f);
    void buildSeepageflowShapeVolume(SeepageflowVolumeData &data,
                                     Vector<float4> shape, float3 color,
                                     float sandDensity, bool offsetY = false,
                                     float worldLowestY = 0.f);
    void buildSeepageflowShapeMultiVolume(SeepageflowMultiVolumeData &data,
                                          Vector<float4> shape, float3 color,
                                          float sandDensity, float3 cda0asat,
                                          float2 amcamcp, bool offsetY = false,
                                          float worldLowestY = 0.f);

    void buildMultiSphRen14Volume(MultiSphRen14VolumeData &data, float3 lowest,
                                  int3 vsize, float particleRadius, float3 color,
                                  float mass, size_t phaseIdx);
    void buildMultiSphYan16Volume(MultiSphYan16VolumeData &data, float3 lowest,
                                  int3 vsize, float particleRadius, float3 color,
                                  float mass, size_t phaseIdx, size_t phaseType);

    inline constexpr bool emitterStatus() const { return bEnable; }

  private:
    bool bEnable;
  };

  typedef SharedPtr<CudaVolumeEmitter> CudaVolumeEmitterPtr;
} // namespace KIRI

#endif