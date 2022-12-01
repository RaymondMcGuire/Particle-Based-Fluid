/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-20 13:23:15
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 15:10:13
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\system\cuda_sph_system.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_pbs_cuda/system/cuda_base_system_gpu.cuh>
#include <kiri_pbs_cuda/system/cuda_sph_system.cuh>
#include <kiri_pbs_cuda/system/cuda_sph_system_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>

namespace KIRI {

CudaSphSystem::CudaSphSystem(CudaSphParticlesPtr &fluid_particles,
                             CudaBoundaryParticlesPtr &boundary_particles,
                             CudaSphSolverPtr &solver,
                             CudaGNSearcherPtr &searcher,
                             CudaGNBoundarySearcherPtr &boundarySearcher,
                             CudaEmitterPtr &emitter,
                             const bool adaptiveSubTimeStep)
    : CudaBaseSystem(std::static_pointer_cast<CudaBaseSolver>(solver), searcher,
                     boundary_particles, boundarySearcher,
                     fluid_particles->maxSize(), adaptiveSubTimeStep),
      mFluids(std::move(fluid_particles)), mEmitter(std::move(emitter)),
      mEmitterCounter(0), mEmitterElapsedTime(0.f),
      mCudaGridSize(CuCeilDiv(fluid_particles->maxSize(), KIRI_CUBLOCKSIZE)) {

  if (CUDA_SPH_EMITTER_PARAMS.enable) {
    switch (CUDA_SPH_EMITTER_PARAMS.emit_type) {
    case CudaSphEmitterType::SQUARE:
      mEmitter->buildSquareEmitter(CUDA_SPH_PARAMS.particle_radius,
                                   CUDA_SPH_EMITTER_PARAMS.emit_radius);
      break;
    case CudaSphEmitterType::CIRCLE:
      mEmitter->buildCircleEmitter(CUDA_SPH_PARAMS.particle_radius,
                                   CUDA_SPH_EMITTER_PARAMS.emit_radius);
      break;
    case CudaSphEmitterType::RECTANGLE:
      mEmitter->buildRectangleEmitter(CUDA_SPH_PARAMS.particle_radius,
                                      CUDA_SPH_EMITTER_PARAMS.emit_width,
                                      CUDA_SPH_EMITTER_PARAMS.emit_height);
      break;
    }
  }

  // compute boundary volume(Akinci2012)
  computeBoundaryVolume();
}

void CudaSphSystem::moveBoundary(const float3 lowestPoint,
                                 const float3 highestPoint) {
  this->updateWorldSize(lowestPoint, highestPoint);
  BoundaryData boundary_data;
  auto boundaryEmitter = std::make_shared<CudaBoundaryEmitter>();

  boundaryEmitter->buildWorldBoundary(
      boundary_data, CUDA_BOUNDARY_PARAMS.lowest_point,
      CUDA_BOUNDARY_PARAMS.highest_point, CUDA_SPH_PARAMS.particle_radius);

  // build boundary searcher
  mBoundaries = std::make_shared<CudaBoundaryParticles>(boundary_data.pos,
                                                        boundary_data.label);
  mBoundarySearcher->buildGNSearcher(mBoundaries);
  computeBoundaryVolume();
}

void CudaSphSystem::onUpdateSolver(float renderInterval) {
  mSearcher->buildGNSearcher(mFluids);
  CudaSphSolverPtr solver;
  if (CUDA_SPH_PARAMS.solver_type == SPH_SOLVER)
    solver = std::dynamic_pointer_cast<CudaSphSolver>(mSolver);
  if (CUDA_SPH_PARAMS.solver_type == MSSPH_SOLVER)
    solver = std::dynamic_pointer_cast<CudaMultiSphSolver>(mSolver);
  else if (CUDA_SPH_PARAMS.solver_type == WCSPH_SOLVER)
    solver = std::dynamic_pointer_cast<CudaWCSphSolver>(mSolver);
  else if (CUDA_SPH_PARAMS.solver_type == IISPH_SOLVER)
    solver = std::dynamic_pointer_cast<CudaIISphSolver>(mSolver);
  else if (CUDA_SPH_PARAMS.solver_type == DFSPH_SOLVER)
    solver = std::dynamic_pointer_cast<CudaDFSphSolver>(mSolver);
  else if (CUDA_SPH_PARAMS.solver_type == PBF_SOLVER)
    solver = std::dynamic_pointer_cast<CudaPBFSolver>(mSolver);

  solver->updateSolver(mFluids, mBoundaries, mSearcher->cellStart(),
                       mBoundarySearcher->cellStart(), renderInterval,
                       CUDA_SPH_PARAMS, CUDA_BOUNDARY_PARAMS);

  // emitter
  if (mEmitter->emitterStatus() && CUDA_SPH_EMITTER_PARAMS.run) {
    if (CUDA_SPH_PARAMS.solver_type == SPH_SOLVER ||
        CUDA_SPH_PARAMS.solver_type == IISPH_SOLVER) {
      size_t numOfStep =
          2.f * CUDA_SPH_PARAMS.particle_radius /
          (length(mEmitter->emitterVelocity()) * CUDA_SPH_PARAMS.dt);
      if (mEmitterCounter++ % numOfStep == 0) {
        auto p = mEmitter->Emit();
        if (mFluids->size() + p.size() < mFluids->maxSize())
          mFluids->appendParticles(
              p, mEmitter->emittRadius(), CUDA_SPH_EMITTER_PARAMS.emit_col,
              mEmitter->emitterVelocity(), CUDA_SPH_PARAMS.rest_mass);
        else
          mEmitter->setEmitterStatus(false);
        // printf("fluid particle number=%zd, max=%zd \n", mFluids->size(),
        //        mFluids->maxSize());
      }
    } else if (CUDA_SPH_PARAMS.solver_type == WCSPH_SOLVER ||
               CUDA_SPH_PARAMS.solver_type == DFSPH_SOLVER) {
      mEmitterElapsedTime +=
          renderInterval / static_cast<float>(this->subTimeStepsNum());
      if ((length(mEmitter->emitterVelocity()) * mEmitterElapsedTime) /
              (2.f * CUDA_SPH_PARAMS.particle_radius) >=
          static_cast<float>(mEmitterCounter)) {
        auto p = mEmitter->Emit();
        if (mFluids->size() + p.size() < mFluids->maxSize())
          mFluids->appendParticles(
              p, mEmitter->emittRadius(), CUDA_SPH_EMITTER_PARAMS.emit_col,
              mEmitter->emitterVelocity(), CUDA_SPH_PARAMS.rest_mass);
        else
          mEmitter->setEmitterStatus(false);
        // printf("fluid particle number=%zd, max=%zd \n", mFluids->size(),
        // mFluids->maxSize());
        mEmitterCounter++;
      }
    }
  }

  cudaDeviceSynchronize();
  KIRI_CUKERNAL();
}

void CudaSphSystem::onTransferGPUData2VBO(float4 *cudaPositionVBO,
                                          float4 *cudaColorVBO) {
  MRP1_GPU2VBO_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      cudaPositionVBO, cudaColorVBO, mFluids->posPtr(), mFluids->colorPtr(),
      mFluids->radiusPtr(), mFluids->size(), 0);

  cudaDeviceSynchronize();
  KIRI_CUKERNAL();
}

void CudaSphSystem::computeBoundaryVolume() {
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
} // namespace KIRI
