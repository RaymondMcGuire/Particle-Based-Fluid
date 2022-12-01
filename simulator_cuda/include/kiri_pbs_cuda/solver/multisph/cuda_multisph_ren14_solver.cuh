/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-11 11:44:38
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-15 17:44:23
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\multisph\cuda_multisph_ren14_solver.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_MULTISPH_REN14_SOLVER_CUH_
#define _CUDA_MULTISPH_REN14_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/cuda_base_solver.cuh>

#include <kiri_pbs_cuda/data/cuda_boundary_params.h>
#include <kiri_pbs_cuda/data/cuda_multisph_params.h>

#include <kiri_pbs_cuda/kernel/cuda_sph_kernel.cuh>

#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_multisph_ren14_particles.cuh>

namespace KIRI
{
  class CudaMultiSphRen14Solver : public CudaBaseSolver
  {
  public:
    explicit CudaMultiSphRen14Solver(const size_t num) : CudaBaseSolver(num) {}

    virtual ~CudaMultiSphRen14Solver() noexcept {}

    virtual void updateSolver(CudaMultiSphRen14ParticlesPtr &particles,
                              CudaBoundaryParticlesPtr &boundaries,
                              const CudaArray<size_t> &cellStart,
                              const CudaArray<size_t> &boundaryCellStart,
                              float timeIntervalInSeconds,
                              CudaMultiSphRen14Params params,
                              CudaBoundaryParams bparams);

  protected:
    virtual void computeRestMixData(CudaMultiSphRen14ParticlesPtr &particles,
                                    const size_t phaseNum);

    virtual void advect(CudaMultiSphRen14ParticlesPtr &particles, const float dt,
                        const float3 lowestPoint, const float3 highestPoint,
                        const float radius);

    virtual void computeMixDensity(CudaMultiSphRen14ParticlesPtr &particles,
                                   CudaBoundaryParticlesPtr &boundaries,
                                   const size_t phaseNum,
                                   const CudaArray<size_t> &cellStart,
                                   const CudaArray<size_t> &boundaryCellStart,
                                   const float3 lowestPoint,
                                   const float kernelRadius, const int3 gridSize);

    virtual void computeMixPressure(CudaMultiSphRen14ParticlesPtr &particles,
                                    const size_t phaseNum, const bool miscible,
                                    const float stiff);

    virtual void computeGradientTerm(CudaMultiSphRen14ParticlesPtr &particles,
                                     const size_t phaseNum, const bool miscible,
                                     const CudaArray<size_t> &cellStart,
                                     const float3 lowestPoint,
                                     const float kernelRadius,
                                     const int3 gridSize);

    virtual void computeDriftVelocities(CudaMultiSphRen14ParticlesPtr &particles,
                                        const size_t phaseNum,
                                        const bool miscible, const float tou,
                                        const float sigma, const float3 gravity);

    virtual void computeDeltaVolumeFraction(
        CudaMultiSphRen14ParticlesPtr &particles, const size_t phaseNum,
        const bool miscible, const CudaArray<size_t> &cellStart,
        const float3 lowestPoint, const float kernelRadius, const int3 gridSize);

    virtual void correctVolumeFraction(
        CudaMultiSphRen14ParticlesPtr &particles,
        const size_t phaseNum,
        const bool miscible,
        const float stiff,
        const float dt);

    virtual void computeMultiSphAcc(CudaMultiSphRen14ParticlesPtr &particles,
                                    CudaBoundaryParticlesPtr &boundaries,
                                    const size_t phaseNum, const float3 gravity,
                                    const float soundSpeed, const float bnu,
                                    const CudaArray<size_t> &cellStart,
                                    const CudaArray<size_t> &boundaryCellStart,
                                    const float3 lowestPoint,
                                    const float kernelRadius,
                                    const int3 gridSize);

    virtual void computeNextTimeStepData(
        CudaMultiSphRen14ParticlesPtr &particles,
        const size_t phaseNum,
        const bool miscible);

  private:
    float mVolumeFractionChangeSpeed = 5.f;
  };

  typedef SharedPtr<CudaMultiSphRen14Solver> CudaMultiSphRen14SolverPtr;
} // namespace KIRI

#endif /* _CUDA_MULTISPH_REN14_SOLVER_CUH_ */