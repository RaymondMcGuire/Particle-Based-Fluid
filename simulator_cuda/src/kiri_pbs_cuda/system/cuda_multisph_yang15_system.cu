/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-12 23:08:31
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 17:50:39
 * @FilePath:
 * \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\system\cuda_multisph_yang15_system.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_pbs_cuda/system/cuda_base_system_gpu.cuh>
#include <kiri_pbs_cuda/system/cuda_multisph_yang15_system.cuh>
#include <kiri_pbs_cuda/system/cuda_sph_system_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>

namespace KIRI {

__global__ void
_InitMultiSphYang15System_CUDA(Yang15PhaseDataBlock1 *phaseDataBlock1,
                               const size_t *phaseLabel, const size_t num,
                               const size_t phaseNum) {
  const size_t i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  for (size_t k = 0; k < phaseNum; ++k)
    if (phaseLabel[i] == k)
      phaseDataBlock1[i].mass_ratios[k] = 1.f;
    else
      phaseDataBlock1[i].mass_ratios[k] = 0.f;
  return;
}

CudaMultiSphYang15System::CudaMultiSphYang15System(
    CudaMultiSphYang15ParticlesPtr &fluid_particles,
    CudaBoundaryParticlesPtr &boundary_particles,
    CudaMultiSphYang15SolverPtr &solver, CudaGNSearcherPtr &searcher,
    CudaGNBoundarySearcherPtr &boundarySearcher, CudaEmitterPtr &emitter,
    bool adaptiveSubTimeStep)
    : CudaBaseSystem(std::static_pointer_cast<CudaBaseSolver>(solver), searcher,
                     boundary_particles, boundarySearcher,
                     fluid_particles->maxSize(), adaptiveSubTimeStep),
      mFluids(std::move(fluid_particles)), mEmitter(std::move(emitter)),
      mEmitterCounter(0), mEmitterElapsedTime(0.f),
      mCudaGridSize(CuCeilDiv(fluid_particles->maxSize(), KIRI_CUBLOCKSIZE)) {

  // compute boundary volume(Akinci2012)
  computeBoundaryVolume();

  // init multisph system
  initMultiSphSystem();
}

void CudaMultiSphYang15System::moveBoundary(const float3 lowestPoint,
                                            const float3 highestPoint) {
  this->updateWorldSize(lowestPoint, highestPoint);
  BoundaryData boundary_data;
  auto boundaryEmitter = std::make_shared<CudaBoundaryEmitter>();

  boundaryEmitter->buildWorldBoundary(
      boundary_data, CUDA_BOUNDARY_PARAMS.lowest_point,
      CUDA_BOUNDARY_PARAMS.highest_point,
      CUDA_MULTISPH_YANG15_PARAMS.particle_radius);

  // build boundary searcher
  mBoundaries = std::make_shared<CudaBoundaryParticles>(boundary_data.pos,
                                                        boundary_data.label);
  mBoundarySearcher->buildGNSearcher(mBoundaries);
  computeBoundaryVolume();
}

void CudaMultiSphYang15System::onUpdateSolver(float renderInterval) {
  mSearcher->buildGNSearcher(mFluids);

  auto solver = std::dynamic_pointer_cast<CudaMultiSphYang15Solver>(mSolver);
  solver->updateSolver(mFluids, mBoundaries, mSearcher->cellStart(),
                       mBoundarySearcher->cellStart(), renderInterval,
                       CUDA_MULTISPH_YANG15_PARAMS, CUDA_BOUNDARY_PARAMS);

  cudaDeviceSynchronize();
  KIRI_CUKERNAL();
}

void CudaMultiSphYang15System::onTransferGPUData2VBO(float4 *cudaPositionVBO,
                                                     float4 *cudaColorVBO) {
  TransferGPUData2VBO_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      cudaPositionVBO, cudaColorVBO, mFluids->posPtr(), mFluids->colorPtr(),
      CUDA_MULTISPH_YANG15_PARAMS.particle_radius, mFluids->size(), 0);

  cudaDeviceSynchronize();
  KIRI_CUKERNAL();
}

void CudaMultiSphYang15System::computeBoundaryVolume() {
  auto mCudaBoundaryGridSize = CuCeilDiv(mBoundaries->size(), KIRI_CUBLOCKSIZE);

  ComputeBoundaryVolume_CUDA<<<mCudaBoundaryGridSize, KIRI_CUBLOCKSIZE>>>(
      mBoundaries->posPtr(), mBoundaries->volumePtr(), mBoundaries->size(),
      mBoundarySearcher->cellStartPtr(), mBoundarySearcher->gridSize(),
      ThrustHelper::Pos2GridXYZ<float3>(mBoundarySearcher->lowestPoint(),
                                        mBoundarySearcher->cellSize(),
                                        mBoundarySearcher->gridSize()),
      ThrustHelper::GridXYZ2GridHash(mBoundarySearcher->gridSize()),
      Poly6Kernel(mBoundarySearcher->cellSize()));
  cudaDeviceSynchronize();
  KIRI_CUKERNAL();
}

void CudaMultiSphYang15System::initMultiSphSystem() {
  auto initCudaGridSize = CuCeilDiv(mFluids->size(), KIRI_CUBLOCKSIZE);
  _InitMultiSphYang15System_CUDA<<<initCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      mFluids->phaseDataBlock1Ptr(), mFluids->phaseLabelPtr(), mFluids->size(),
      CUDA_MULTISPH_YANG15_PARAMS.phase_num);

  cudaDeviceSynchronize();
  KIRI_CUKERNAL();
}

} // namespace KIRI
