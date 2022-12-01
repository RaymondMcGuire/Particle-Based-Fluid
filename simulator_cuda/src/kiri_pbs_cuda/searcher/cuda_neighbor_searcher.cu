/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-11 11:44:38
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 17:02:42
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\searcher\cuda_neighbor_searcher.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>

#include <kiri_pbs_cuda/particle/cuda_dfsph_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_iisph_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_sph_particles.cuh>

#include <kiri_pbs_cuda/particle/cuda_multisph_ren14_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_multisph_yang15_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_multisph_yan16_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_pbf_particles.cuh>

namespace KIRI
{

  CudaGNBaseSearcher::CudaGNBaseSearcher(
      const float3 lowestPoint,
      const float3 highestPoint,
      const size_t maxNumOfParticles,
      const float cellSize)
      : mLowestPoint(lowestPoint),
        mHighestPoint(highestPoint),
        mCellSize(cellSize),
        mGridSize(make_int3((highestPoint - lowestPoint) / cellSize)),
        mNumOfGridCells(mGridSize.x * mGridSize.y * mGridSize.z + 1),
        mCellStart(mNumOfGridCells),
        mMaxNumOfParticles(maxNumOfParticles),
        mGridIdxArray(max(mNumOfGridCells, maxNumOfParticles)),
        mCudaGridSize(CuCeilDiv(maxNumOfParticles, KIRI_CUBLOCKSIZE)) {}

  void CudaGNBaseSearcher::buildGNSearcher(const CudaParticlesPtr &particles)
  {
    thrust::transform(
        thrust::device, particles->posPtr(),
        particles->posPtr() + particles->size(), particles->particle2CellPtr(),
        ThrustHelper::Pos2GridHash<float3>(mLowestPoint, mCellSize, mGridSize));

    this->sortData(particles);

    thrust::fill(thrust::device, mCellStart.data(),
                 mCellStart.data() + mNumOfGridCells, 0);
    _CountingInCell_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        mCellStart.data(), particles->particle2CellPtr(), particles->size());
    thrust::exclusive_scan(thrust::device, mCellStart.data(),
                           mCellStart.data() + mNumOfGridCells,
                           mCellStart.data());

    cudaDeviceSynchronize();
    KIRI_CUKERNAL();
  }

  void CudaGNBaseSearcher::updateWorldSize(const float3 lowestPoint, const float3 highestPoint)
  {
    mLowestPoint = lowestPoint;
    mHighestPoint = highestPoint;
    mGridSize = make_int3((highestPoint - lowestPoint) / mCellSize);
    mNumOfGridCells = mGridSize.x * mGridSize.y * mGridSize.z + 1;

    mCellStart.resize(mNumOfGridCells);
    mGridIdxArray.resize(max(mNumOfGridCells, mMaxNumOfParticles));
  }

  CudaGNSearcher::CudaGNSearcher(const float3 lowestPoint, const float3 highestPoint,
                                 const size_t num, const float cellSize,
                                 const SearcherParticleType type)
      : CudaGNBaseSearcher(lowestPoint, highestPoint, num, cellSize), mSearcherParticleType(type) {}

  void CudaGNSearcher::sortData(const CudaParticlesPtr &particles)
  {

    auto particle_size = particles->size();

    if (mSearcherParticleType == SearcherParticleType::SPH)
    {
      auto fluids = std::dynamic_pointer_cast<CudaSphParticles>(particles);

      KIRI_CUCALL(cudaMemcpy(mGridIdxArray.data(), fluids->particle2CellPtr(),
                             sizeof(size_t) * particle_size,
                             cudaMemcpyDeviceToDevice));
      thrust::sort_by_key(thrust::device, mGridIdxArray.data(),
                          mGridIdxArray.data() + particle_size, fluids->posPtr());

      KIRI_CUCALL(cudaMemcpy(mGridIdxArray.data(), fluids->particle2CellPtr(),
                             sizeof(size_t) * particle_size,
                             cudaMemcpyDeviceToDevice));
      thrust::sort_by_key(thrust::device, mGridIdxArray.data(),
                          mGridIdxArray.data() + particle_size, fluids->velPtr());

      KIRI_CUCALL(cudaMemcpy(mGridIdxArray.data(), fluids->particle2CellPtr(),
                             sizeof(size_t) * particle_size,
                             cudaMemcpyDeviceToDevice));
      thrust::sort_by_key(thrust::device, mGridIdxArray.data(),
                          mGridIdxArray.data() + particle_size,
                          fluids->colorPtr());

      KIRI_CUCALL(cudaMemcpy(mGridIdxArray.data(), fluids->particle2CellPtr(),
                             sizeof(size_t) * particle_size,
                             cudaMemcpyDeviceToDevice));
      thrust::sort_by_key(thrust::device, mGridIdxArray.data(),
                          mGridIdxArray.data() + particle_size,
                          fluids->massPtr());

      KIRI_CUCALL(cudaMemcpy(mGridIdxArray.data(), fluids->particle2CellPtr(),
                             sizeof(size_t) * particle_size,
                             cudaMemcpyDeviceToDevice));
      thrust::sort_by_key(thrust::device, mGridIdxArray.data(),
                          mGridIdxArray.data() + particle_size,
                          fluids->radiusPtr());
    }
    else if (mSearcherParticleType == SearcherParticleType::IISPH)
    {
      auto fluids = std::dynamic_pointer_cast<CudaIISphParticles>(particles);

      KIRI_CUCALL(cudaMemcpy(mGridIdxArray.data(), fluids->particle2CellPtr(),
                             sizeof(size_t) * particle_size,
                             cudaMemcpyDeviceToDevice));
      thrust::sort_by_key(thrust::device, mGridIdxArray.data(),
                          mGridIdxArray.data() + particle_size,
                          thrust::make_zip_iterator(thrust::make_tuple(
                              fluids->posPtr(), fluids->velPtr(),
                              fluids->colorPtr(), fluids->GetLastPressurePtr())));
    }
    else if (mSearcherParticleType == SearcherParticleType::DFSPH)
    {
      auto fluids = std::dynamic_pointer_cast<CudaDFSphParticles>(particles);

      KIRI_CUCALL(cudaMemcpy(mGridIdxArray.data(), fluids->particle2CellPtr(),
                             sizeof(size_t) * particle_size,
                             cudaMemcpyDeviceToDevice));
      thrust::sort_by_key(
          thrust::device, mGridIdxArray.data(),
          mGridIdxArray.data() + particle_size,
          thrust::make_zip_iterator(thrust::make_tuple(
              fluids->posPtr(), fluids->velPtr(), fluids->colorPtr(),
              fluids->massPtr(), fluids->warmStiffPtr())));
    }
    else if (mSearcherParticleType == SearcherParticleType::PBF)
    {
      auto fluids = std::dynamic_pointer_cast<CudaPBFParticles>(particles);

      KIRI_CUCALL(cudaMemcpy(mGridIdxArray.data(), fluids->particle2CellPtr(),
                             sizeof(size_t) * particle_size,
                             cudaMemcpyDeviceToDevice));
      thrust::sort_by_key(
          thrust::device, mGridIdxArray.data(),
          mGridIdxArray.data() + particle_size,
          thrust::make_zip_iterator(thrust::make_tuple(
              fluids->posPtr(), fluids->velPtr(), fluids->colorPtr(),
              fluids->lastPosPtr(), fluids->oldPosPtr())));
    }
    else if (mSearcherParticleType == SearcherParticleType::MULTISPH_REN14)
    {
      auto fluids =
          std::dynamic_pointer_cast<CudaMultiSphRen14Particles>(particles);

      KIRI_CUCALL(cudaMemcpy(mGridIdxArray.data(), fluids->particle2CellPtr(),
                             sizeof(size_t) * particle_size,
                             cudaMemcpyDeviceToDevice));
      thrust::sort_by_key(
          thrust::device, mGridIdxArray.data(),
          mGridIdxArray.data() + particle_size,
          thrust::make_zip_iterator(thrust::make_tuple(
              fluids->posPtr(),
              fluids->velPtr(),
              fluids->colorPtr(),
              fluids->mixMassPtr())));

      KIRI_CUCALL(cudaMemcpy(mGridIdxArray.data(), fluids->particle2CellPtr(),
                             sizeof(size_t) * particle_size,
                             cudaMemcpyDeviceToDevice));
      thrust::sort_by_key(thrust::device, mGridIdxArray.data(),
                          mGridIdxArray.data() + particle_size,
                          fluids->phaseDataBlock1Ptr());

      KIRI_CUCALL(cudaMemcpy(mGridIdxArray.data(), fluids->particle2CellPtr(),
                             sizeof(size_t) * particle_size,
                             cudaMemcpyDeviceToDevice));
      thrust::sort_by_key(thrust::device, mGridIdxArray.data(),
                          mGridIdxArray.data() + particle_size,
                          fluids->phaseDataBlock2Ptr());
    }
    else if (mSearcherParticleType == SearcherParticleType::MULTISPH_YANG15)
    {
      auto fluids =
          std::dynamic_pointer_cast<CudaMultiSphYang15Particles>(particles);

      KIRI_CUCALL(cudaMemcpy(mGridIdxArray.data(), fluids->particle2CellPtr(),
                             sizeof(size_t) * particle_size,
                             cudaMemcpyDeviceToDevice));
      thrust::sort_by_key(
          thrust::device, mGridIdxArray.data(),
          mGridIdxArray.data() + particle_size,
          thrust::make_zip_iterator(thrust::make_tuple(
              fluids->posPtr(),
              fluids->velPtr(),
              fluids->colorPtr(),
              fluids->mixMassPtr(),
              fluids->lastPosPtr(),
              fluids->oldPosPtr())));

      KIRI_CUCALL(cudaMemcpy(mGridIdxArray.data(), fluids->particle2CellPtr(),
                             sizeof(size_t) * particle_size,
                             cudaMemcpyDeviceToDevice));
      thrust::sort_by_key(thrust::device, mGridIdxArray.data(),
                          mGridIdxArray.data() + particle_size,
                          fluids->phaseDataBlock1Ptr());

      KIRI_CUCALL(cudaMemcpy(mGridIdxArray.data(), fluids->particle2CellPtr(),
                             sizeof(size_t) * particle_size,
                             cudaMemcpyDeviceToDevice));
      thrust::sort_by_key(thrust::device, mGridIdxArray.data(),
                          mGridIdxArray.data() + particle_size,
                          fluids->phaseDataBlock2Ptr());
    }


    cudaDeviceSynchronize();
    KIRI_CUKERNAL();
  }

  CudaGNBoundarySearcher::CudaGNBoundarySearcher(const float3 lowestPoint, const float3 highestPoint,
                                                 const size_t num,
                                                 const float cellSize)
      : CudaGNBaseSearcher(lowestPoint, highestPoint, num, cellSize) {}

  void CudaGNBoundarySearcher::sortData(const CudaParticlesPtr &particles)
  {
    auto boundaries = std::dynamic_pointer_cast<CudaBoundaryParticles>(particles);

    KIRI_CUCALL(cudaMemcpy(mGridIdxArray.data(), boundaries->particle2CellPtr(),
                           sizeof(size_t) * particles->size(),
                           cudaMemcpyDeviceToDevice));
    thrust::sort_by_key(thrust::device, mGridIdxArray.data(),
                        mGridIdxArray.data() + particles->size(),
                        thrust::make_zip_iterator(thrust::make_tuple(
                            boundaries->posPtr(), boundaries->labelPtr())));

    cudaDeviceSynchronize();
    KIRI_CUKERNAL();
  }

} // namespace KIRI
