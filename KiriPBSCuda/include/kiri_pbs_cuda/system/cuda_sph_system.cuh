/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 22:52:09
 * @LastEditTime: 2021-02-10 16:33:10
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\system\cuda_sph_system.cuh
 */
#ifndef _CUDA_SPH_SYSTEM_CUH_
#define _CUDA_SPH_SYSTEM_CUH_

#pragma once

#include <kiri_pbs_cuda/kernel/cuda_sph_kernel.cuh>
#include <kiri_pbs_cuda/cuda_base_solver.cuh>
#include <kiri_pbs_cuda/particle/cuda_sph_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>

namespace KIRI
{
    class CudaSphSystem
    {
    public:
        CudaSphSystem(
            CudaSphParticlesPtr &fluidParticles,
            CudaBoundaryParticlesPtr &boundaryParticles,
            CudaBaseSolverPtr &solver,
            CudaGNSearcherPtr &searcher,
            CudaGNBoundarySearcherPtr &boundarySearcher,
            bool openGL = true);

        CudaSphSystem(const CudaSphSystem &) = delete;
        CudaSphSystem &operator=(const CudaSphSystem &) = delete;

        void UpdateSystemForVBO();
        float UpdateSystem();

        int Size() const { return FluidSize(); }
        int FluidSize() const { return (*mFluids).Size(); }

        int TotalSize() const { return (*mFluids).Size(); }

        auto GetFluids() const { return static_cast<const SharedPtr<CudaSphParticles>>(mFluids); }

        inline uint PositionsVBO() const { return mPositionsVBO; }
        inline uint ColorsVBO() const { return mColorsVBO; }

        ~CudaSphSystem() noexcept {}

    private:
        CudaSphParticlesPtr mFluids;
        CudaBoundaryParticlesPtr mBoundaries;
        CudaBaseSolverPtr mSolver;
        CudaGNSearcherPtr mSearcher;
        CudaGNBoundarySearcherPtr mBoundarySearcher;

        bool bOpenGL;

        const int mCudaGridSize;

        float4 *pptr, *cptr;

        // VBO for OpenGL
        uint mPositionsVBO;
        uint mColorsVBO;
        struct cudaGraphicsResource *mCudaGraphPosVBORes, *mCudaGraphColorVBORes;

        void NeighborSearch(
            const CudaSphParticlesPtr &fluids,
            CudaArray<int> &cellStart);

        void CopyGPUData2VBO(
            float4 *pos,
            float4 *col,
            const CudaSphParticlesPtr &fluids);

        void ComputeBoundaryVolume();
    };

    typedef SharedPtr<CudaSphSystem> CudaSphSystemPtr;
} // namespace KIRI

#endif