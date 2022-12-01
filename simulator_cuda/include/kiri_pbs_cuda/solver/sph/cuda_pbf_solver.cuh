/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-12 14:47:42
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-20 10:53:36
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sph\cuda_pbf_solver.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_PBF_SOLVER_CUH_
#define _CUDA_PBF_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_pbf_particles.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver.cuh>

namespace KIRI
{
  class CudaPBFSolver final : public CudaSphSolver
  {
  public:
    virtual void updateSolver(CudaSphParticlesPtr &fluids,
                              CudaBoundaryParticlesPtr &boundaries,
                              const CudaArray<size_t> &cellStart,
                              const CudaArray<size_t> &boundaryCellStart,
                              float timeIntervalInSeconds, CudaSphParams params,
                              CudaBoundaryParams bparams) override;

    explicit CudaPBFSolver(const size_t num, const bool incompressible)
        : CudaSphSolver(num),
          mAvgDensityError(0.f),
          mIncompressiable(incompressible),
          mCorrK(0.05f),
          mCorrN(4.f),
          mLambdaEps(1000.f),
          mDeltaQ(0.3f),
          mVisc(0.1f),
          mBoundaryVisc(0.f),
          mVorticityCoeff(0.01f) {}

    virtual ~CudaPBFSolver() noexcept {}

  private:
    float mAvgDensityError, mVisc, mBoundaryVisc, mVorticityCoeff;

    bool mIncompressiable;

    // incompressiable
    const size_t MIN_ITERATION = 2;
    const size_t MAX_ITERATION = 100;
    const float MAX_DENSITY_ERROR = 0.01f;

    // tensile instability
    float mDeltaQ, mCorrK, mCorrN, mLambdaEps;
    const size_t MAX_REALTIME_ITERATION = 4;

    virtual void computeVorticityConfinement(
        CudaPBFParticlesPtr &fluids,
        const float vorticityCoeff,
        const CudaArray<size_t> &cellStart,
        const float3 lowestPoint,
        const float kernelRadius, const int3 gridSize);

    virtual void computeViscosityXSPH(
        CudaPBFParticlesPtr &fluids,
        CudaBoundaryParticlesPtr &boundaries,
        const float rho0,
        const float visc,
        const float boundaryVisc,
        const float dt,
        const CudaArray<size_t> &cellStart,
        const CudaArray<size_t> &boundaryCellStart,
        const float3 lowestPoint,
        const float kernelRadius,
        const int3 gridSize);

    virtual void constaintProjection(
        CudaPBFParticlesPtr &fluids,
        CudaBoundaryParticlesPtr &boundaries,
        const float rho0,
        const float dt,
        const CudaArray<size_t> &cellStart,
        const CudaArray<size_t> &boundaryCellStart,
        const float3 lowestPoint,
        const float kernelRadius,
        const int3 gridSize);

    virtual void computeLagrangeMultiplier(
        CudaPBFParticlesPtr &fluids,
        CudaBoundaryParticlesPtr &boundaries,
        const float rho0,
        const CudaArray<size_t> &cellStart,
        const CudaArray<size_t> &boundaryCellStart,
        const float3 lowestPoint,
        const float kernelRadius,
        const int3 gridSize);

    virtual void solveDensityConstrain(
        CudaPBFParticlesPtr &fluids,
        CudaBoundaryParticlesPtr &boundaries,
        const float rho0,
        const float dt,
        const CudaArray<size_t> &cellStart,
        const CudaArray<size_t> &boundaryCellStart,
        const float3 lowestPoint,
        const float kernelRadius,
        const int3 gridSize);
  };

  typedef SharedPtr<CudaPBFSolver> CudaPBFSolverPtr;
} // namespace KIRI

#endif /* _CUDA_PBF_SOLVER_CUH_ */