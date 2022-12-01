/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-12 23:06:52
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 17:50:59
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\system\cuda_multisph_yang15_system.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_MULTISPH_YANG15_SYSTEM_CUH_
#define _CUDA_MULTISPH_YANG15_SYSTEM_CUH_

#pragma once

#include <kiri_pbs_cuda/system/cuda_base_system.cuh>

#include <kiri_pbs_cuda/particle/cuda_multisph_yang15_particles.cuh>
#include <kiri_pbs_cuda/solver/multisph/cuda_multisph_yang15_solver.cuh>

#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>

#include <kiri_pbs_cuda/emitter/cuda_emitter.cuh>
namespace KIRI
{
  class CudaMultiSphYang15System : public CudaBaseSystem
  {
  public:
    explicit CudaMultiSphYang15System(
        CudaMultiSphYang15ParticlesPtr &fluidParticles,
        CudaBoundaryParticlesPtr &boundaryParticles,
        CudaMultiSphYang15SolverPtr &solver,
        CudaGNSearcherPtr &searcher,
        CudaGNBoundarySearcherPtr &boundarySearcher,
        CudaEmitterPtr &emitter,
        bool adaptiveSubTimeStep = false);

    CudaMultiSphYang15System(const CudaMultiSphYang15System &) = delete;
    CudaMultiSphYang15System &operator=(const CudaMultiSphYang15System &) = delete;
    virtual ~CudaMultiSphYang15System() noexcept {}

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
    CudaMultiSphYang15ParticlesPtr mFluids;
    CudaEmitterPtr mEmitter;

    void computeBoundaryVolume();
    void initMultiSphSystem();
  };

  typedef SharedPtr<CudaMultiSphYang15System> CudaMultiSphYang15SystemPtr;
} // namespace KIRI

#endif