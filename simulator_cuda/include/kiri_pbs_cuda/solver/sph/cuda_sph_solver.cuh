/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 19:05:14
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-20 11:55:09
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sph\cuda_sph_solver.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_SPH_SOLVER_CUH_
#define _CUDA_SPH_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/cuda_base_solver.cuh>

#include <kiri_pbs_cuda/data/cuda_sph_params.h>

#include <kiri_pbs_cuda/kernel/cuda_sph_kernel.cuh>

#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_sph_particles.cuh>

namespace KIRI
{
  class CudaSphSolver : public CudaBaseSolver
  {
  public:
    explicit CudaSphSolver(const size_t num) : CudaBaseSolver(num) {}

    virtual ~CudaSphSolver() noexcept {}

    virtual void updateSolver(CudaSphParticlesPtr &fluids,
                              CudaBoundaryParticlesPtr &boundaries,
                              const CudaArray<size_t> &cellStart,
                              const CudaArray<size_t> &boundaryCellStart,
                              float timeIntervalInSeconds, CudaSphParams params,
                              CudaBoundaryParams bparams);

  protected:
    virtual void extraForces(CudaSphParticlesPtr &fluids, const float3 gravity);

    virtual void advect(CudaSphParticlesPtr &fluids, const float dt,
                        const float3 lowestPoint, const float3 highestPoint,
                        const float radius);

    virtual void computeDensity(CudaSphParticlesPtr &fluids,
                                CudaBoundaryParticlesPtr &boundaries,
                                const float rho0,
                                const CudaArray<size_t> &cellStart,
                                const CudaArray<size_t> &boundaryCellStart,
                                const float3 lowestPoint,
                                const float kernelRadius, const int3 gridSize);

    virtual void computeNablaTerm(CudaSphParticlesPtr &fluids,
                                  CudaBoundaryParticlesPtr &boundaries,
                                  const CudaArray<size_t> &cellStart,
                                  const CudaArray<size_t> &boundaryCellStart,
                                  const float3 lowestPoint,
                                  const float kernelRadius, const int3 gridSize,
                                  const float rho0, const float stiff);

    virtual void computeViscosityTerm(CudaSphParticlesPtr &fluids,
                                      CudaBoundaryParticlesPtr &boundaries,
                                      const CudaArray<size_t> &cellStart,
                                      const CudaArray<size_t> &boundaryCellStart,
                                      const float rho0, const float visc,
                                      const float bnu, const float3 lowestPoint,
                                      const float kernelRadius,
                                      const int3 gridSize);

    virtual void computeArtificialViscosityTerm(
        CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
        const CudaArray<size_t> &cellStart,
        const CudaArray<size_t> &boundaryCellStart, const float rho0,
        const float nu, const float bnu, const float3 lowestPoint,
        const float kernelRadius, const int3 gridSize);

    void computeAkinci13Normal(CudaSphParticlesPtr &fluids,
                               const CudaArray<size_t> &cellStart,
                               const float3 lowestPoint, const float kernelRadius,
                               const int3 gridSize);

    void computeAkinci13Term(CudaSphParticlesPtr &fluids,
                             CudaBoundaryParticlesPtr &boundaries,
                             const CudaArray<size_t> &cellStart,
                             const CudaArray<size_t> &boundaryCellStart,
                             const float rho0, const float beta,
                             const float gamma, const float3 lowestPoint,
                             const float kernelRadius, const int3 gridSize);
  };

  typedef SharedPtr<CudaSphSolver> CudaSphSolverPtr;
} // namespace KIRI

#endif /* _CUDA_SPH_SOLVER_CUH_ */