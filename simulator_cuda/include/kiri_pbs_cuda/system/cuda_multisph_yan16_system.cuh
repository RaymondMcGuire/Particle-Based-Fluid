/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 19:05:14
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 14:42:27
 * @FilePath: \KiriPBSCuda\include\kiri_pbs_cuda\system\cuda_multisph_yan16_system.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_MULTISPH_YAN16_SYSTEM_CUH_
#define _CUDA_MULTISPH_YAN16_SYSTEM_CUH_

#pragma once

#include <kiri_pbs_cuda/system/cuda_base_system.cuh>

#include <kiri_pbs_cuda/particle/cuda_multisph_yan16_particles.cuh>
#include <kiri_pbs_cuda/solver/multisph/cuda_multisph_yan16_solver.cuh>

#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>

#include <kiri_pbs_cuda/emitter/cuda_emitter.cuh>
namespace KIRI
{
  class CudaMultiSphYan16System : public CudaBaseSystem
  {
  public:
    explicit CudaMultiSphYan16System(
        CudaMultiSphYan16ParticlesPtr &fluid_particles,
        CudaBoundaryParticlesPtr &boundary_particles,
        CudaMultiSphYan16SolverPtr &solver,
        CudaGNSearcherPtr &searcher,
        CudaGNBoundarySearcherPtr &boundarySearcher,
        CudaEmitterPtr &emitter,
        bool adaptiveSubTimeStep = false);

    CudaMultiSphYan16System(const CudaMultiSphYan16System &) = delete;
    CudaMultiSphYan16System &operator=(const CudaMultiSphYan16System &) = delete;
    virtual ~CudaMultiSphYan16System() noexcept {}

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
    CudaMultiSphYan16ParticlesPtr mFluids;
    CudaEmitterPtr mEmitter;

    void computeBoundaryVolume();
    void initMultiSphSystem();
  };

  typedef SharedPtr<CudaMultiSphYan16System> CudaMultiSphYan16SystemPtr;
} // namespace KIRI

#endif