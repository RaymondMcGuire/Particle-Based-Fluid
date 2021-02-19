/*
 * @Author: Xu.WANG
 * @Date: 2021-02-05 12:33:37
 * @LastEditTime: 2021-02-10 00:10:11
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\searcher\cuda_neighbor_searcher.cu
 */

#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher_gpu.cuh>

namespace KIRI
{

    CudaGNBaseSearcher::CudaGNBaseSearcher(
        const float3 lowestPoint,
        const float3 highestPoint,
        const uint numOfParticles,
        const float cellSize)
        : mLowestPoint(lowestPoint),
          mHighestPoint(highestPoint),
          mCellSize(cellSize),
          mGridSize(make_int3((highestPoint - lowestPoint) / cellSize)),
          mNumOfGridCells(mGridSize.x * mGridSize.y * mGridSize.z + 1),
          mCellStart(mNumOfGridCells),
          mNumOfParticles(numOfParticles),
          mGridIdxArray(max(mNumOfGridCells, mNumOfParticles)),
          mCudaGridSize(CuCeilDiv(numOfParticles, KIRI_CUBLOCKSIZE))
    {
    }

    void CudaGNBaseSearcher::BuildGNSearcher(const CudaParticlesPtr &particles)
    {

        thrust::transform(thrust::device,
                          particles->GetPosPtr(), particles->GetPosPtr() + mNumOfParticles,
                          mGridIdxArray.Data(),
                          ThrustHelper::Pos2GridHash<float3>(mLowestPoint, mCellSize, mGridSize));

        this->SortData(particles);

        thrust::fill(thrust::device, mCellStart.Data(), mCellStart.Data() + mNumOfGridCells, 0);
        CountingInCell_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(mCellStart.Data(), mGridIdxArray.Data(), mNumOfParticles);
        thrust::exclusive_scan(thrust::device, mCellStart.Data(), mCellStart.Data() + mNumOfGridCells, mCellStart.Data());

        KIRI_CUKERNAL();
    }

    CudaGNSearcher::CudaGNSearcher(
        const float3 lp,
        const float3 hp,
        const uint num,
        const float cellSize)
        : CudaGNBaseSearcher(lp, hp, num, cellSize) {}

    void CudaGNSearcher::SortData(const CudaParticlesPtr &particles)
    {
        auto fluids = std::dynamic_pointer_cast<CudaSphParticles>(particles);
        thrust::sort_by_key(thrust::device,
                            mGridIdxArray.Data(),
                            mGridIdxArray.Data() + mNumOfParticles,
                            thrust::make_zip_iterator(
                                thrust::make_tuple(
                                    fluids->GetPosPtr(),
                                    fluids->GetVelPtr(),
                                    fluids->GetColPtr())));
    }

    CudaGNBoundarySearcher::CudaGNBoundarySearcher(
        const float3 lp,
        const float3 hp,
        const uint num,
        const float cellSize)
        : CudaGNBaseSearcher(lp, hp, num, cellSize) {}

    void CudaGNBoundarySearcher::SortData(const CudaParticlesPtr &particles)
    {
        auto boundaries = std::dynamic_pointer_cast<CudaBoundaryParticles>(particles);
        thrust::sort_by_key(thrust::device,
                            mGridIdxArray.Data(),
                            mGridIdxArray.Data() + mNumOfParticles,
                            boundaries->GetPosPtr());
    }

} // namespace KIRI
