/*
 * @Author: Xu.WANG
 * @Date: 2020-07-26 17:30:04
 * @LastEditTime: 2021-02-08 16:24:20
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\searcher\cuda_neighbor_searcher.cuh
 */

#ifndef _CUDA_NEIGHBOR_SEARCHER_CUH_
#define _CUDA_NEIGHBOR_SEARCHER_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_sph_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>

namespace KIRI
{

    class CudaGNBaseSearcher
    {
    public:
        explicit CudaGNBaseSearcher(
            const float3 lowestPoint,
            const float3 highestPoint,
            const uint numOfParticles,
            const float cellSize);

        CudaGNBaseSearcher(const CudaGNBaseSearcher &) = delete;
        CudaGNBaseSearcher &operator=(const CudaGNBaseSearcher &) = delete;

        virtual ~CudaGNBaseSearcher() noexcept {}

        float3 GetLowestPoint() const { return mLowestPoint; }
        float3 GetHighestPoint() const { return mHighestPoint; }
        float GetCellSize() const { return mCellSize; }
        int3 GetGridSize() const { return mGridSize; }

        uint *GetCellStartPtr() const { return mCellStart.Data(); }
        const CudaArray<uint> &GetCellStart() const { return mCellStart; }

        uint *GetGridIdxArrayPtr() const { return mGridIdxArray.Data(); }
        const CudaArray<uint> &GetGridIdxArray() const { return mGridIdxArray; }

        void BuildGNSearcher(const CudaParticlesPtr &particles);

    protected:
        const uint mCudaGridSize;
        const int3 mGridSize;
        const float mCellSize;
        const float3 mLowestPoint;
        const float3 mHighestPoint;
        const uint mNumOfGridCells;
        const uint mNumOfParticles;

        CudaArray<uint> mGridIdxArray;
        CudaArray<uint> mCellStart;

        virtual void SortData(const CudaParticlesPtr &particles) = 0;
    };

    class CudaGNSearcher final : public CudaGNBaseSearcher
    {
    public:
        explicit CudaGNSearcher(
            const float3 lp,
            const float3 hp,
            const uint num,
            const float cellSize);

        CudaGNSearcher(const CudaGNSearcher &) = delete;
        CudaGNSearcher &operator=(const CudaGNSearcher &) = delete;

        virtual ~CudaGNSearcher() noexcept {}

    protected:
        virtual void SortData(const CudaParticlesPtr &particles) override final;
    };

    class CudaGNBoundarySearcher final : public CudaGNBaseSearcher
    {
    public:
        explicit CudaGNBoundarySearcher(
            const float3 lp,
            const float3 hp,
            const uint num,
            const float cellSize);

        CudaGNBoundarySearcher(const CudaGNBoundarySearcher &) = delete;
        CudaGNBoundarySearcher &operator=(const CudaGNBoundarySearcher &) = delete;

        virtual ~CudaGNBoundarySearcher() noexcept {}

    protected:
        virtual void SortData(const CudaParticlesPtr &particles) override final;
    };

    typedef SharedPtr<CudaGNBaseSearcher> CudaGNBaseSearcherPtr;
    typedef SharedPtr<CudaGNSearcher> CudaGNSearcherPtr;
    typedef SharedPtr<CudaGNBoundarySearcher> CudaGNBoundarySearcherPtr;
} // namespace KIRI

#endif /* CudaNeighborSearcher */