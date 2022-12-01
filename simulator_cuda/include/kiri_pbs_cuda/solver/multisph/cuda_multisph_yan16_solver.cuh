/*
 * @Author: Xu.WANG
 * @Date: 2021-02-01 14:31:30
 * @LastEditTime: 2021-04-23 02:13:38
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\multisph\cuda_multisph_yan16_solver.cuh
 */

#ifndef _CUDA_MULTISPH_YAN16_SOLVER_CUH_
#define _CUDA_MULTISPH_YAN16_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/cuda_base_solver.cuh>

#include <kiri_pbs_cuda/data/cuda_boundary_params.h>
#include <kiri_pbs_cuda/data/cuda_multisph_params.h>

#include <kiri_pbs_cuda/kernel/cuda_sph_kernel.cuh>

#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_multisph_yan16_particles.cuh>

namespace KIRI
{
  class CudaMultiSphYan16Solver : public CudaBaseSolver
  {
  public:
    explicit CudaMultiSphYan16Solver(const size_t num) : CudaBaseSolver(num) {}

    virtual ~CudaMultiSphYan16Solver() noexcept {}

    virtual void updateSolver(CudaMultiSphYan16ParticlesPtr &particles,
                              CudaBoundaryParticlesPtr &boundaries,
                              const CudaArray<size_t> &cellStart,
                              const CudaArray<size_t> &boundaryCellStart,
                              float timeIntervalInSeconds,
                              CudaMultiSphYan16Params params,
                              CudaBoundaryParams bparams);

  protected:
    virtual void advect(CudaMultiSphYan16ParticlesPtr &particles, const float dt,
                        const float3 lowestPoint, const float3 highestPoint,
                        const float radius);

    virtual void computeMixDensity(CudaMultiSphYan16ParticlesPtr &particles,
                                   CudaBoundaryParticlesPtr &boundaries,
                                   const size_t phaseNum,
                                   const CudaArray<size_t> &cellStart,
                                   const CudaArray<size_t> &boundaryCellStart,
                                   const float3 lowestPoint,
                                   const float kernelRadius, const int3 gridSize);

    virtual void computeMixPressure(CudaMultiSphYan16ParticlesPtr &particles,
                                    const size_t phaseNum, const bool miscible,
                                    const float stiff);

    virtual void computeGradientTerm(CudaMultiSphYan16ParticlesPtr &particles,
                                     const size_t phaseNum, const bool miscible,
                                     const CudaArray<size_t> &cellStart,
                                     const float3 lowestPoint,
                                     const float kernelRadius,
                                     const int3 gridSize);

    virtual void computeDriftVelocities(CudaMultiSphYan16ParticlesPtr &particles,
                                        const size_t phaseNum,
                                        const bool miscible, const float tou,
                                        const float sigma, const float3 gravity);

    virtual void computeDeltaVolumeFraction(
        CudaMultiSphYan16ParticlesPtr &particles, const size_t phaseNum,
        const bool miscible, const CudaArray<size_t> &cellStart,
        const float3 lowestPoint, const float kernelRadius, const int3 gridSize);

    virtual void correctVolumeFraction(CudaMultiSphYan16ParticlesPtr &particles,
                                       const size_t phaseNum, const bool miscible,
                                       const float dt);

    virtual void ComputeDeviatoricStressRateTensor(
        CudaMultiSphYan16ParticlesPtr &particles, const float G,
        const size_t phaseNum, const CudaArray<size_t> &cellStart,
        const float3 lowestPoint, const float kernelRadius, const int3 gridSize);

    virtual void
    CorrectDeviatoricStressTensor(CudaMultiSphYan16ParticlesPtr &particles,
                                  const size_t phaseNum, const float Y);

    virtual void computeMultiSphAcc(CudaMultiSphYan16ParticlesPtr &particles,
                                    CudaBoundaryParticlesPtr &boundaries,
                                    const size_t phaseNum, const float3 gravity,
                                    const float soundSpeed, const float bnu,
                                    const CudaArray<size_t> &cellStart,
                                    const CudaArray<size_t> &boundaryCellStart,
                                    const float3 lowestPoint,
                                    const float kernelRadius,
                                    const int3 gridSize);

    virtual void computeNextTimeStepData(CudaMultiSphYan16ParticlesPtr &particles,
                                         const size_t phaseNum);
  };

  typedef SharedPtr<CudaMultiSphYan16Solver> CudaMultiSphYan16SolverPtr;
} // namespace KIRI

#endif /* _CUDA_MULTISPH_YAN16_SOLVER_CUH_ */