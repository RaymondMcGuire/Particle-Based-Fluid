/*
 * @Author: Xu.WANG
 * @Date: 2021-02-01 14:31:30
 * @LastEditTime: 2021-02-14 20:23:10
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\cuda_base_solver.cuh
 */

#ifndef _CUDA_BASE_SOLVER_CUH_
#define _CUDA_BASE_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/data/cuda_sph_params.h>
#include <kiri_pbs_cuda/data/cuda_boundary_params.h>
#include <kiri_pbs_cuda/particle/cuda_sph_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>

namespace KIRI
{
    class CudaBaseSolver
    {
    public:
        virtual void UpdateSolver(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<uint> &cellStart,
            const CudaArray<uint> &boundaryCellStart,
            CudaSphParams params,
            CudaBoundaryParams bparams) = 0;

        virtual ~CudaBaseSolver() noexcept {}

    protected:
        virtual void Advect(
            CudaSphParticlesPtr &fluids,
            const float dt,
            const float3 lowestPoint,
            const float3 highestPoint,
            const float radius) = 0;

        virtual void ExtraForces(
            CudaSphParticlesPtr &fluids,
            const float3 gravity) = 0;
    };

    typedef SharedPtr<CudaBaseSolver> CudaBaseSolverPtr;
} // namespace KIRI

#endif