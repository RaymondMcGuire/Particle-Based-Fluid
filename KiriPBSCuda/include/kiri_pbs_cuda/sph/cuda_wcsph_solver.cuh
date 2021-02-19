/*
 * @Author: Xu.WANG
 * @Date: 2021-02-01 14:31:30
 * @LastEditTime: 2021-02-15 14:03:21
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\sph\cuda_wcsph_solver.cuh
 */

#ifndef _CUDA_WCSPH_SOLVER_CUH_
#define _CUDA_WCSPH_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/sph/cuda_sph_solver.cuh>

namespace KIRI
{
    class CudaWCSphSolver final : public CudaSphSolver
    {
    public:
        virtual void UpdateSolver(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<uint> &cellStart,
            const CudaArray<uint> &boundaryCellStart,
            CudaSphParams params,
            CudaBoundaryParams bparams) override;

        explicit CudaWCSphSolver(
            const uint num, const float negativeScale = 0.f)
            : CudaSphSolver(num), mNegativeScale(negativeScale)
        {
        }

        virtual ~CudaWCSphSolver() noexcept {}

    private:
        float mNegativeScale;
        bool bCubicKernel = false;

        virtual void ComputeDensity(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const float rho0,
            const CudaArray<uint> &cellStart,
            const CudaArray<uint> &boundaryCellStart,
            const float3 lowestPoint,
            const float kernelSize,
            const int3 gridSize) override;

        virtual void ComputeNablaTerm(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<uint> &cellStart,
            const CudaArray<uint> &boundaryCellStart,
            const float3 lowestPoint,
            const float kernelSize,
            const int3 gridSize,
            const float rho0,
            const float stiff) override;

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
            const int3 gridSize) override;

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
            const int3 gridSize) override;
    };

    typedef SharedPtr<CudaWCSphSolver> CudaWCSphSolverPtr;
} // namespace KIRI

#endif /* _CUDA_WCSPH_SOLVER_CUH_ */