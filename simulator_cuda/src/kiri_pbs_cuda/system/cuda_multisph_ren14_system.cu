/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 19:05:14
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-12 13:02:17
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\system\cuda_multisph_ren14_system.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_pbs_cuda/system/cuda_base_system_gpu.cuh>
#include <kiri_pbs_cuda/system/cuda_multisph_ren14_system.cuh>
#include <kiri_pbs_cuda/system/cuda_sph_system_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>

namespace KIRI
{

  static __global__ void InitMultiSphSolver_CUDA(Ren14PhaseDataBlock2 *phaseDataBlock2,
                                                 const size_t *phaseLabel,
                                                 const size_t num,
                                                 const size_t phaseNum)
  {
    const size_t i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    for (size_t k = 0; k < phaseNum; ++k)
      if (phaseLabel[i] == k)
        phaseDataBlock2[i].volume_fractions[k] = 1.f;
      else
        phaseDataBlock2[i].volume_fractions[k] = 0.f;
    return;
  }

  CudaMultiSphRen14System::CudaMultiSphRen14System(
      CudaMultiSphRen14ParticlesPtr &fluid_particles,
      CudaBoundaryParticlesPtr &boundary_particles,
      CudaMultiSphRen14SolverPtr &solver, CudaGNSearcherPtr &searcher,
      CudaGNBoundarySearcherPtr &boundarySearcher, CudaEmitterPtr &emitter,
      bool adaptiveSubTimeStep)
      : CudaBaseSystem(
            std::static_pointer_cast<CudaBaseSolver>(solver),
            searcher,
            boundary_particles,
            boundarySearcher,
            fluid_particles->maxSize(),
            adaptiveSubTimeStep),
        mFluids(std::move(fluid_particles)),
        mEmitter(std::move(emitter)),
        mEmitterCounter(0),
        mEmitterElapsedTime(0.f),
        mCudaGridSize(CuCeilDiv(fluid_particles->maxSize(), KIRI_CUBLOCKSIZE))
  {

    // compute boundary volume(Akinci2012)
    computeBoundaryVolume();

    // init multisph system
    initMultiSphSystem();
  }

  void CudaMultiSphRen14System::onUpdateSolver(float renderInterval)
  {
    mSearcher->buildGNSearcher(mFluids);

    auto solver = std::dynamic_pointer_cast<CudaMultiSphRen14Solver>(mSolver);
    solver->updateSolver(mFluids, mBoundaries, mSearcher->cellStart(),
                         mBoundarySearcher->cellStart(), renderInterval,
                         CUDA_MULTISPH_REN14_PARAMS, CUDA_BOUNDARY_PARAMS);

    cudaDeviceSynchronize();
    KIRI_CUKERNAL();
  }

  void CudaMultiSphRen14System::onTransferGPUData2VBO(float4 *cudaPositionVBO,
                                                      float4 *cudaColorVBO)
  {
    TransferGPUData2VBO_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        cudaPositionVBO, cudaColorVBO, mFluids->posPtr(), mFluids->colorPtr(),
        CUDA_MULTISPH_REN14_PARAMS.particle_radius, mFluids->size(), 0);

    cudaDeviceSynchronize();
    KIRI_CUKERNAL();
  }

  void CudaMultiSphRen14System::computeBoundaryVolume()
  {
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

  void CudaMultiSphRen14System::initMultiSphSystem()
  {
    auto initCudaGridSize = CuCeilDiv(mFluids->size(), KIRI_CUBLOCKSIZE);
    InitMultiSphSolver_CUDA<<<initCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        mFluids->phaseDataBlock2Ptr(),
        mFluids->phaseLabelPtr(),
        mFluids->size(),
        CUDA_MULTISPH_REN14_PARAMS.phase_num);

    cudaDeviceSynchronize();
    KIRI_CUKERNAL();
  }

} // namespace KIRI
