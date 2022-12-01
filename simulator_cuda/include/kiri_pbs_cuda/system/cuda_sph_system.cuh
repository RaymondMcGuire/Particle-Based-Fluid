/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-11 11:44:38
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 16:00:24
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\system\cuda_sph_system.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_SPH_SYSTEM_CUH_
#define _CUDA_SPH_SYSTEM_CUH_

#pragma once

#include <kiri_pbs_cuda/system/cuda_base_system.cuh>

#include <kiri_pbs_cuda/kernel/cuda_sph_kernel.cuh>
#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_sph_particles.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>

#include <kiri_pbs_cuda/solver/sph/cuda_dfsph_solver.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_iisph_solver.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_multi_sph_solver.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_pbf_solver.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_wcsph_solver.cuh>

#include <kiri_pbs_cuda/emitter/cuda_emitter.cuh>

namespace KIRI
{
  class CudaSphSystem : public CudaBaseSystem
  {
  public:
    explicit CudaSphSystem(
        CudaSphParticlesPtr &fluid_particles,
        CudaBoundaryParticlesPtr &boundary_particles,
        CudaSphSolverPtr &solver,
        CudaGNSearcherPtr &searcher,
        CudaGNBoundarySearcherPtr &boundarySearcher,
        CudaEmitterPtr &emitter,
        bool adaptiveSubTimeStep = false);

    CudaSphSystem(const CudaSphSystem &) = delete;
    CudaSphSystem &operator=(const CudaSphSystem &) = delete;
    virtual ~CudaSphSystem() noexcept {}

    inline size_t fluidSize() const { return (*mFluids).size(); }
    inline size_t numOfParticles() const { return (*mFluids).size(); }
    inline size_t maxNumOfParticles() const { return (*mFluids).maxSize(); }

    void moveBoundary(const float3 lowestPoint, const float3 highestPoint);

  protected:
    virtual void onUpdateSolver(float timeIntervalInSeconds) override;
    virtual void onTransferGPUData2VBO(float4 *cudaPositionVBO,
                                       float4 *cudaColorVBO) override;

  private:
    const size_t mCudaGridSize;
    size_t mEmitterCounter;
    float mEmitterElapsedTime;
    CudaSphParticlesPtr mFluids;
    CudaEmitterPtr mEmitter;

    void computeBoundaryVolume();
  };

  typedef SharedPtr<CudaSphSystem> CudaSphSystemPtr;
} // namespace KIRI

#endif