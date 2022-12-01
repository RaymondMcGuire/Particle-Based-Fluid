/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-25 12:33:06
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-07-26 12:26:33
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sph\cuda_dfsph_solver.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_DFSPH_SOLVER_CUH_
#define _CUDA_DFSPH_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver.cuh>

namespace KIRI
{
  class CudaDFSphSolver final : public CudaSphSolver
  {
  public:
    explicit CudaDFSphSolver(const size_t num, const float dt = 0.001f,
                             const size_t pressureMinIter = 2,
                             const size_t pressureMaxIter = 100,
                             const size_t divergenceMinIter = 1,
                             const size_t divergenceMaxIter = 100,
                             const float pressureMaxError = 1e-3f,
                             const float divergenceMaxError = 1e-3f)
        : CudaSphSolver(num), mDt(dt), mPressureErrorThreshold(pressureMaxError),
          mDivergenceErrorThreshold(divergenceMaxError),
          mPressureMinIter(pressureMinIter), mPressureMaxIter(pressureMaxIter),
          mDivergenceMinIter(divergenceMinIter),
          mDivergenceMaxIter(divergenceMaxIter) {}

    virtual ~CudaDFSphSolver() noexcept {}

    virtual void updateSolver(CudaSphParticlesPtr &fluids,
                              CudaBoundaryParticlesPtr &boundaries,
                              const CudaArray<size_t> &cellStart,
                              const CudaArray<size_t> &boundaryCellStart,
                              float timeIntervalInSeconds, CudaSphParams params,
                              CudaBoundaryParams bparams) override;

  protected:
    virtual void computeDensity(CudaSphParticlesPtr &fluids,
                                CudaBoundaryParticlesPtr &boundaries,
                                const float rho0,
                                const CudaArray<size_t> &cellStart,
                                const CudaArray<size_t> &boundaryCellStart,
                                const float3 lowestPoint,
                                const float kernelRadius, const int3 gridSize) override;

  private:
    float mDt;
    float mDivergenceErrorThreshold, mPressureErrorThreshold;
    size_t mPressureMinIter, mPressureMaxIter, mDivergenceMinIter,
        mDivergenceMaxIter;

    const float CFL_FACTOR = 1.f;
    const float CFL_MIN_TIMESTEP_SIZE = 0.0001f;
    const float CFL_MAX_TIMESTEP_SIZE = 0.005f;

    void velAdvect(CudaSphParticlesPtr &fluids);

    virtual void advect(CudaSphParticlesPtr &fluids, const float dt,
                        const float3 lowestPoint, const float3 highestPoint,
                        const float radius) override;

    void computeTimeStepsByCFL(CudaSphParticlesPtr &fluids,
                               const float particleRadius,
                               const float timeIntervalInSeconds);

    void computeAlpha(CudaSphParticlesPtr &fluids,
                      CudaBoundaryParticlesPtr &boundaries, const float rho0,
                      const CudaArray<size_t> &cellStart,
                      const CudaArray<size_t> &boundaryCellStart,
                      const float3 lowestPoint, const float kernelRadius,
                      const int3 gridSize);

    size_t divergenceSolver(CudaSphParticlesPtr &fluids,
                            CudaBoundaryParticlesPtr &boundaries,
                            const float rho0, const CudaArray<size_t> &cellStart,
                            const CudaArray<size_t> &boundaryCellStart,
                            const float3 lowestPoint, const float kernelRadius,
                            const int3 gridSize);

    size_t pressureSolver(CudaSphParticlesPtr &fluids,
                          CudaBoundaryParticlesPtr &boundaries, const float rho0,
                          const CudaArray<size_t> &cellStart,
                          const CudaArray<size_t> &boundaryCellStart,
                          const float3 lowestPoint, const float kernelRadius,
                          const int3 gridSize);
  };

  typedef SharedPtr<CudaDFSphSolver> CudaDFSphSolverPtr;
} // namespace KIRI

#endif /* _CUDA_DFSPH_SOLVER_CUH_ */