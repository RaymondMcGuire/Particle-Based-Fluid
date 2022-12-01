/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 19:05:14
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-15 18:07:46
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sph\cuda_iisph_solver.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_IISPH_SOLVER_CUH_
#define _CUDA_IISPH_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver.cuh>

namespace KIRI
{
  class CudaIISphSolver final : public CudaSphSolver
  {
  public:
    explicit CudaIISphSolver(const size_t num, const size_t minIter = 2,
                             const size_t maxIter = 100,
                             const float maxError = 0.01f)
        : CudaSphSolver(num), mMaxError(maxError), mMinIter(minIter),
          mMaxIter(maxIter) {}

    virtual ~CudaIISphSolver() noexcept {}

    virtual void updateSolver(CudaSphParticlesPtr &fluids,
                              CudaBoundaryParticlesPtr &boundaries,
                              const CudaArray<size_t> &cellStart,
                              const CudaArray<size_t> &boundaryCellStart,
                              float timeIntervalInSeconds, CudaSphParams params,
                              CudaBoundaryParams bparams) override;

  private:
    float mMaxError;
    size_t mMinIter, mMaxIter;

    virtual void advect(CudaSphParticlesPtr &fluids, const float dt,
                        const float3 lowestPoint, const float3 highestPoint,
                        const float radius) override;

    void predictVelAdvect(CudaSphParticlesPtr &fluids, const float dt);

    void computeDiiTerm(CudaSphParticlesPtr &fluids,
                        CudaBoundaryParticlesPtr &boundaries,
                        const CudaArray<size_t> &cellStart,
                        const CudaArray<size_t> &boundaryCellStart,
                        const float rho0, const float3 lowestPoint,
                        const float kernelRadius, const int3 gridSize);

    void computeAiiTerm(CudaSphParticlesPtr &fluids,
                        CudaBoundaryParticlesPtr &boundaries,
                        const CudaArray<size_t> &cellStart,
                        const CudaArray<size_t> &boundaryCellStart,
                        const float rho0, const float dt,
                        const float3 lowestPoint, const float kernelRadius,
                        const int3 gridSize);

    size_t pressureSolver(CudaSphParticlesPtr &fluids,
                          CudaBoundaryParticlesPtr &boundaries, const float rho0,
                          const float dt, const CudaArray<size_t> &cellStart,
                          const CudaArray<size_t> &boundaryCellStart,
                          const float3 lowestPoint, const float kernelRadius,
                          const int3 gridSize);

    void computePressureAcceleration(CudaSphParticlesPtr &fluids,
                                     CudaBoundaryParticlesPtr &boundaries,
                                     const CudaArray<size_t> &cellStart,
                                     const CudaArray<size_t> &boundaryCellStart,
                                     const float rho0, const float3 lowestPoint,
                                     const float kernelRadius,
                                     const int3 gridSize);
  };

  typedef SharedPtr<CudaIISphSolver> CudaIISphSolverPtr;
} // namespace KIRI

#endif /* _CUDA_IISPH_SOLVER_CUH_ */