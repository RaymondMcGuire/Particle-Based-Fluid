/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 19:05:14
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-15 18:07:03
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sph\cuda_multi_sph_solver.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_MULTI_SPH_SOLVER_CUH_
#define _CUDA_MULTI_SPH_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver.cuh>

namespace KIRI
{
    class CudaMultiSphSolver : public CudaSphSolver
    {
    public:
        explicit CudaMultiSphSolver(
            const size_t num)
            : CudaSphSolver(num)
        {
        }

        virtual ~CudaMultiSphSolver() noexcept {}

        virtual void updateSolver(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            float timeIntervalInSeconds,
            CudaSphParams params,
            CudaBoundaryParams bparams) override;

    protected:
        virtual void advectMRSph(
            CudaSphParticlesPtr &fluids,
            const float dt,
            const float3 lowestPoint,
            const float3 highestPoint);

        virtual void computeMRDensity(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const float rho0,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);

        virtual void computeMRNablaTerm(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize,
            const float rho0,
            const float stiff);

        virtual void computeMRViscosityTerm(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float rho0,
            const float visc,
            const float bnu,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);

        virtual void computeMRArtificialViscosityTerm(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float rho0,
            const float nu,
            const float bnu,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);
    };

    typedef SharedPtr<CudaMultiSphSolver> CudaMultiSphSolverPtr;
} // namespace KIRI

#endif /* _CUDA_MULTI_SPH_SOLVER_CUH_ */