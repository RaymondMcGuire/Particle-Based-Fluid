/*
 * @Author: Xu.WANG
 * @Date: 2021-02-01 14:31:30
 * @LastEditTime: 2021-02-15 14:07:59
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\sph\cuda_sph_solver.cuh
 */

#ifndef _CUDA_SPH_SOLVER_CUH_
#define _CUDA_SPH_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/kernel/cuda_sph_kernel.cuh>
#include <kiri_pbs_cuda/cuda_base_solver.cuh>

namespace KIRI
{
    class CudaSphSolver : public CudaBaseSolver
    {
    public:
        virtual void UpdateSolver(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<uint> &cellStart,
            const CudaArray<uint> &boundaryCellStart,
            CudaSphParams params,
            CudaBoundaryParams bparams) override;

        explicit CudaSphSolver(
            const uint num)
            : mCudaGridSize(CuCeilDiv(num, KIRI_CUBLOCKSIZE))
        {
        }

        virtual ~CudaSphSolver() noexcept {}

    protected:
        uint mCudaGridSize;

        virtual void ExtraForces(
            CudaSphParticlesPtr &fluids,
            const float3 gravity) override final;

        virtual void Advect(
            CudaSphParticlesPtr &fluids,
            const float dt,
            const float3 lowestPoint,
            const float3 highestPoint,
            const float radius) override final;

    protected:
        virtual void ComputeDensity(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const float rho0,
            const CudaArray<uint> &cellStart,
            const CudaArray<uint> &boundaryCellStart,
            const float3 lowestPoint,
            const float kernelSize,
            const int3 gridSize);

        virtual void ComputeNablaTerm(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<uint> &cellStart,
            const CudaArray<uint> &boundaryCellStart,
            const float3 lowestPoint,
            const float kernelSize,
            const int3 gridSize,
            const float rho0,
            const float stiff);

        virtual void ComputeViscosityTerm(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<uint> &cellStart,
            const CudaArray<uint> &boundaryCellStart,
            const float rho0,
            const float visc,
            const float bnu,
            const float3 lowestPoint,
            const float kernelSize,
            const int3 gridSize);

        virtual void ComputeArtificialViscosityTerm(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<uint> &cellStart,
            const CudaArray<uint> &boundaryCellStart,
            const float rho0,
            const float nu,
            const float bnu,
            const float3 lowestPoint,
            const float kernelSize,
            const int3 gridSize);
    };

    typedef SharedPtr<CudaSphSolver> CudaSphSolverPtr;
} // namespace KIRI

#endif /* _CUDA_SPH_SOLVER_CUH_ */