/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-27 10:50:17
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 14:02:17
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\searcher\cuda_neighbor_searcher.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_NEIGHBOR_SEARCHER_CUH_
#define _CUDA_NEIGHBOR_SEARCHER_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>
namespace KIRI
{

  enum SearcherParticleType
  {
    SPH = 0,
    IISPH = 1,
    DEM = 2,
    MRDEM = 3,
    BNSPH = 4,
    MULTISPH_REN14 = 5,
    MULTISPH_YAN16 = 6,
    SEEPAGE = 7,
    SEEPAGE_MULTI = 8,
    IISEEPAGE = 9,
    PBF = 10,
    PBFBN = 11,
    DFSPH = 12,
    MULTISPH_YANG15 = 13
  };

  class CudaGNBaseSearcher
  {
  public:
    explicit CudaGNBaseSearcher(
        const float3 lowestPoint,
        const float3 highestPoint,
        const size_t numOfParticles,
        const float cellSize);

    CudaGNBaseSearcher(const CudaGNBaseSearcher &) = delete;
    CudaGNBaseSearcher &operator=(const CudaGNBaseSearcher &) = delete;

    virtual ~CudaGNBaseSearcher() noexcept {}

    float3 lowestPoint() const { return mLowestPoint; }
    float3 highestPoint() const { return mHighestPoint; }
    float cellSize() const { return mCellSize; }
    int3 gridSize() const { return mGridSize; }

    size_t *cellStartPtr() const { return mCellStart.data(); }
    const CudaArray<size_t> &cellStart() const { return mCellStart; }

    size_t *gridIdxArrayPtr() const { return mGridIdxArray.data(); }
    const CudaArray<size_t> &gridIdxArray() const { return mGridIdxArray; }

    void buildGNSearcher(const CudaParticlesPtr &particles);

    void updateWorldSize(const float3 lowestPoint, const float3 highestPoint);

  protected:
    size_t mCudaGridSize;
    int3 mGridSize;
    float mCellSize;
    float3 mLowestPoint;
    float3 mHighestPoint;
    size_t mNumOfGridCells;
    size_t mMaxNumOfParticles;

    CudaArray<size_t> mGridIdxArray;
    CudaArray<size_t> mCellStart;

    virtual void sortData(const CudaParticlesPtr &particles) = 0;
  };

  class CudaGNSearcher final : public CudaGNBaseSearcher
  {
  public:
    explicit CudaGNSearcher(const float3 lowestPoint, const float3 highestPoint, const size_t num,
                            const float cellSize,
                            const SearcherParticleType type);

    CudaGNSearcher(const CudaGNSearcher &) = delete;
    CudaGNSearcher &operator=(const CudaGNSearcher &) = delete;

    virtual ~CudaGNSearcher() noexcept {}

    inline constexpr SearcherParticleType GetSearcherType() const
    {
      return mSearcherParticleType;
    }

  protected:
    virtual void sortData(const CudaParticlesPtr &particles) override final;

  private:
    SearcherParticleType mSearcherParticleType;
  };

  class CudaGNBoundarySearcher final : public CudaGNBaseSearcher
  {
  public:
    explicit CudaGNBoundarySearcher(const float3 lowestPoint, const float3 highestPoint,
                                    const size_t num, const float cellSize);

    CudaGNBoundarySearcher(const CudaGNBoundarySearcher &) = delete;
    CudaGNBoundarySearcher &operator=(const CudaGNBoundarySearcher &) = delete;

    virtual ~CudaGNBoundarySearcher() noexcept {}

  protected:
    virtual void sortData(const CudaParticlesPtr &particles) override final;
  };

  typedef SharedPtr<CudaGNBaseSearcher> CudaGNBaseSearcherPtr;
  typedef SharedPtr<CudaGNSearcher> CudaGNSearcherPtr;
  typedef SharedPtr<CudaGNBoundarySearcher> CudaGNBoundarySearcherPtr;
} // namespace KIRI

#endif /* CudaNeighborSearcher */