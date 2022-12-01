/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 19:05:14
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 14:42:12
 * @FilePath: \KiriPBSCuda\include\kiri_pbs_cuda\system\cuda_multisph_ren14_system.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_MULTISPH_REN14_SYSTEM_CUH_
#define _CUDA_MULTISPH_REN14_SYSTEM_CUH_

#pragma once

#include <kiri_pbs_cuda/system/cuda_base_system.cuh>

#include <kiri_pbs_cuda/particle/cuda_multisph_ren14_particles.cuh>
#include <kiri_pbs_cuda/solver/multisph/cuda_multisph_ren14_solver.cuh>

#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>

#include <kiri_pbs_cuda/emitter/cuda_emitter.cuh>
namespace KIRI
{
  class CudaMultiSphRen14System : public CudaBaseSystem
  {
  public:
    explicit CudaMultiSphRen14System(
        CudaMultiSphRen14ParticlesPtr &fluid_particles,
        CudaBoundaryParticlesPtr &boundary_particles,
        CudaMultiSphRen14SolverPtr &solver,
        CudaGNSearcherPtr &searcher,
        CudaGNBoundarySearcherPtr &boundarySearcher,
        CudaEmitterPtr &emitter,
        bool adaptiveSubTimeStep = false);

    CudaMultiSphRen14System(const CudaMultiSphRen14System &) = delete;
    CudaMultiSphRen14System &operator=(const CudaMultiSphRen14System &) = delete;
    virtual ~CudaMultiSphRen14System() noexcept {}

    inline size_t fluidSize() const { return (*mFluids).size(); }
    inline size_t numOfParticles() const { return (*mFluids).size(); }
    inline size_t maxNumOfParticles() const { return (*mFluids).maxSize(); }

  protected:
    virtual void onUpdateSolver(float timeIntervalInSeconds) override;
    virtual void onTransferGPUData2VBO(float4 *cudaPositionVBO,
                                       float4 *cudaColorVBO) override;

  private:
    const size_t mCudaGridSize;
    size_t mEmitterCounter;
    float mEmitterElapsedTime;
    CudaMultiSphRen14ParticlesPtr mFluids;
    CudaEmitterPtr mEmitter;

    void computeBoundaryVolume();
    void initMultiSphSystem();
  };

  typedef SharedPtr<CudaMultiSphRen14System> CudaMultiSphRen14SystemPtr;
} // namespace KIRI

#endif