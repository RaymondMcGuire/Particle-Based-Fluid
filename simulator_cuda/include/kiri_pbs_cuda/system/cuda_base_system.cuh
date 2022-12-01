/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 19:05:14
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 15:59:20
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\system\cuda_base_system.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_BASE_SYSTEM_CUH_
#define _CUDA_BASE_SYSTEM_CUH_

#pragma once
#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>
#include <kiri_pbs_cuda/solver/cuda_base_solver.cuh>
#include <kiri_pbs_cuda/emitter/cuda_boundary_emitter.cuh>
namespace KIRI
{
  class CudaBaseSystem
  {
  public:
    explicit CudaBaseSystem(
        CudaBaseSolverPtr &solver,
        CudaGNSearcherPtr &searcher,
        CudaBoundaryParticlesPtr &boundary_particles,
        CudaGNBoundarySearcherPtr &boundarySearcher,
        const size_t maxNumOfParticles,
        const bool adaptiveSubTimeStep);

    CudaBaseSystem(const CudaBaseSystem &) = delete;
    CudaBaseSystem &operator=(const CudaBaseSystem &) = delete;
    virtual ~CudaBaseSystem() noexcept {}

    inline size_t colorsVBO() const { return mColorsVBO; }
    inline size_t positionsVBO() const { return mPositionsVBO; }

    inline bool adaptiveSubTimeStep() const { return bAdaptiveSubTimeStep; }
    inline size_t subTimeStepsNum() const { return mSolver->subTimeStepsNum(); }

    float updateSystem(float timeIntervalInSeconds);
    void updateSystemForVBO(float timeIntervalInSeconds);
    void updateWorldSize(const float3 lowestPoint, const float3 highestPoint);

  protected:
    CudaBaseSolverPtr mSolver;
    CudaGNSearcherPtr mSearcher;
    CudaBoundaryParticlesPtr mBoundaries;
    CudaGNBoundarySearcherPtr mBoundarySearcher;

    virtual void onUpdateSolver(float timeIntervalInSeconds) = 0;
    virtual void onTransferGPUData2VBO(float4 *cudaPositionVBO,
                                       float4 *cudaColorVBO) = 0;

  private:
    bool bAdaptiveSubTimeStep;

    float4 *mCudaPositionVBO, *mCudaColorVBO;

    // VBO for OpenGL
    uint mPositionsVBO;
    uint mColorsVBO;
    struct cudaGraphicsResource *mCudaGraphPosVBORes, *mCudaGraphColorVBORes;
  };

  typedef SharedPtr<CudaBaseSystem> CudaBaseSystemPtr;
} // namespace KIRI

#endif