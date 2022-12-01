/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-12 23:28:15
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 17:54:25
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\multisph\cuda_multisph_yang15_solver.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_MULTISPH_YANG15_SOLVER_CUH_
#define _CUDA_MULTISPH_YANG15_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/cuda_base_solver.cuh>

#include <kiri_pbs_cuda/data/cuda_multisph_params.h>

#include <kiri_pbs_cuda/kernel/cuda_sph_kernel.cuh>

#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_multisph_yang15_particles.cuh>

namespace KIRI
{
    class CudaMultiSphYang15Solver : public CudaBaseSolver
    {
    public:
        explicit CudaMultiSphYang15Solver(const size_t num)
            : CudaBaseSolver(num),
              mAvgDensityError(0.f) {}

        virtual ~CudaMultiSphYang15Solver() noexcept {}

        virtual void updateSolver(
            CudaMultiSphYang15ParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            float timeIntervalInSeconds,
            CudaMultiSphYang15Params params,
            CudaBoundaryParams bparams);

    protected:
        virtual void extraForces(
            CudaMultiSphYang15ParticlesPtr &particles,
            const float3 gravity);

        virtual void advect(
            CudaMultiSphYang15ParticlesPtr &particles,
            const float dt,
            const float3 lowestPoint,
            const float3 highestPoint,
            const float radius);

        virtual void computeAggregateDensity(
            CudaMultiSphYang15ParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const size_t phaseNum,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);

        virtual void computeNSCHModel(
            CudaMultiSphYang15ParticlesPtr &particles,
            const size_t phaseNum,
            const float eta,
            const float mobilities,
            const float alpha,
            const float s1,
            const float s2,
            const float epsilon,
            const CudaArray<size_t> &cellStart,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);

        virtual void constaintProjection(
            CudaMultiSphYang15ParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const float dt,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);

        virtual void computeLagrangeMultiplier(
            CudaMultiSphYang15ParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);

        virtual void solveDensityConstrain(
            CudaMultiSphYang15ParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const float dt,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);

        virtual void computeViscosityXSPH(
            CudaMultiSphYang15ParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const float visc,
            const float boundaryVisc,
            const float dt,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);

        void computeSurfaceTension(
            CudaMultiSphYang15ParticlesPtr &particles,
            const float sigma,
            const float eta,
            const float epsilon,
            const float dt,
            const size_t phaseNum,
            const CudaArray<size_t> &cellStart,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);

        virtual void computeNextTimeStepData(
            CudaMultiSphYang15ParticlesPtr &particles,
            const size_t phaseNum);

    private:
        float mAvgDensityError;
        const size_t MIN_ITERATION = 2;
        const size_t MAX_ITERATION = 100;
        const float MAX_DENSITY_ERROR = 0.01f;
    };

    typedef SharedPtr<CudaMultiSphYang15Solver> CudaMultiSphYang15SolverPtr;
} // namespace KIRI

#endif /* _CUDA_MULTISPH_YANG15_SOLVER_CUH_ */