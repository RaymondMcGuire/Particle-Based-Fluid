#include <kiri_pbs_cuda/system/cuda_base_system_gpu.cuh>
#include <kiri_pbs_cuda/system/cuda_multisph_yan16_system.cuh>
#include <kiri_pbs_cuda/system/cuda_sph_system_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>

namespace KIRI
{

  static __global__ void
  InitMultiSphYan16Solver_CUDA(
      Yan16PhaseData *phaseDataBlock1,
      const size_t *phaseLabel,
      const size_t *phaseType,
      const size_t *restPhaseTypes,
      const size_t num,
      const size_t phaseNum)
  {
    const size_t i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    phaseDataBlock1[i].phase_type = phaseType[i];
    phaseDataBlock1[i].last_phase_type = phaseType[i];

    for (size_t k = 0; k < phaseNum; ++k)
    {
      phaseDataBlock1[i].phase_types[k] = restPhaseTypes[k];
      phaseDataBlock1[i].stress_tensor[k] = make_tensor3x3(make_float3(0.f));
      phaseDataBlock1[i].deviatoric_stress_tensor[k] = make_tensor3x3(make_float3(0.f));

      if (phaseLabel[i] == k)
        phaseDataBlock1[i].volume_fractions[k] = 1.f;
      else
        phaseDataBlock1[i].volume_fractions[k] = 0.f;
    }

    return;
  }

  CudaMultiSphYan16System::CudaMultiSphYan16System(
      CudaMultiSphYan16ParticlesPtr &fluid_particles,
      CudaBoundaryParticlesPtr &boundary_particles,
      CudaMultiSphYan16SolverPtr &solver,
      CudaGNSearcherPtr &searcher,
      CudaGNBoundarySearcherPtr &boundarySearcher,
      CudaEmitterPtr &emitter,
      bool adaptiveSubTimeStep)
      : CudaBaseSystem(
            std::static_pointer_cast<CudaBaseSolver>(solver),
            searcher,
            boundary_particles,
            boundarySearcher,
            fluid_particles->maxSize(),
            adaptiveSubTimeStep),
        mFluids(std::move(fluid_particles)),
        mEmitter(std::move(emitter)), mEmitterCounter(0),
        mEmitterElapsedTime(0.f),
        mCudaGridSize(CuCeilDiv(fluid_particles->maxSize(), KIRI_CUBLOCKSIZE))
  {

    // if (CUDA_SPH_EMITTER_PARAMS.enable)
    // {
    //     switch (CUDA_SPH_EMITTER_PARAMS.emit_type)
    //     {
    //     case CudaSphEmitterType::SQUARE:
    //         mEmitter->buildSquareEmitter(
    //             CUDA_MULTISPH_YAN16_PARAMS.particle_radius,
    //             CUDA_SPH_EMITTER_PARAMS.emit_radius);
    //         break;
    //     case CudaSphEmitterType::CIRCLE:
    //         mEmitter->buildCircleEmitter(
    //             CUDA_MULTISPH_YAN16_PARAMS.particle_radius,
    //             CUDA_SPH_EMITTER_PARAMS.emit_radius);
    //         break;
    //     case CudaSphEmitterType::RECTANGLE:
    //         mEmitter->buildRectangleEmitter(
    //             CUDA_MULTISPH_YAN16_PARAMS.particle_radius,
    //             CUDA_SPH_EMITTER_PARAMS.emit_width,
    //             CUDA_SPH_EMITTER_PARAMS.emit_height);
    //         break;
    //     }
    // }

    // compute boundary volume(Akinci2012)
    computeBoundaryVolume();

    // init multisph system
    initMultiSphSystem();
  }

  void CudaMultiSphYan16System::onUpdateSolver(float renderInterval)
  {
    mSearcher->buildGNSearcher(mFluids);
    // CudaSphSolverPtr solver;
    // if (CUDA_MULTISPH_YAN16_PARAMS.solver_type == SPH_SOLVER)
    //     solver = std::dynamic_pointer_cast<CudaSphSolver>(mSolver);
    // else if (CUDA_MULTISPH_YAN16_PARAMS.solver_type == WCSPH_SOLVER)
    //     solver = std::dynamic_pointer_cast<CudaWCSphSolver>(mSolver);

    auto solver = std::dynamic_pointer_cast<CudaMultiSphYan16Solver>(mSolver);
    solver->updateSolver(mFluids, mBoundaries, mSearcher->cellStart(),
                         mBoundarySearcher->cellStart(), renderInterval,
                         CUDA_MULTISPH_YAN16_PARAMS, CUDA_BOUNDARY_PARAMS);

    // emitter
    // if (mEmitter->emitterStatus() && CUDA_SPH_EMITTER_PARAMS.run)
    // {
    //     if (CUDA_MULTISPH_YAN16_PARAMS.solver_type == SPH_SOLVER)
    //     {
    //         size_t numOfStep = 2.f * CUDA_MULTISPH_YAN16_PARAMS.particle_radius
    //         / (length(mEmitter->emitterVelocity()) *
    //         CUDA_MULTISPH_YAN16_PARAMS.dt); if (mEmitterCounter++ % numOfStep
    //         == 0)
    //         {
    //             auto p = mEmitter->Emit();
    //             if (mFluids->size() + p.size() < mFluids->maxSize())
    //                 mFluids->appendParticles(p,
    //                 CUDA_SPH_EMITTER_PARAMS.emit_col,
    //                 mEmitter->emitterVelocity(),
    //                 CUDA_MULTISPH_YAN16_PARAMS.rest_mass);
    //             else
    //                 mEmitter->setEmitterStatus(false);
    //             //printf("fluid particle number=%zd, max=%zd \n",
    //             mFluids->size(), mFluids->maxSize());
    //         }
    //     }
    //     else if (CUDA_MULTISPH_YAN16_PARAMS.solver_type == WCSPH_SOLVER)
    //     {
    //         mEmitterElapsedTime += renderInterval /
    //         static_cast<float>(this->subTimeStepsNum()); if
    //         ((length(mEmitter->emitterVelocity()) * mEmitterElapsedTime) /
    //         (2.f * CUDA_MULTISPH_YAN16_PARAMS.particle_radius) >=
    //         static_cast<float>(mEmitterCounter))
    //         {
    //             auto p = mEmitter->Emit();
    //             if (mFluids->size() + p.size() < mFluids->maxSize())
    //                 mFluids->appendParticles(p,
    //                 CUDA_SPH_EMITTER_PARAMS.emit_col,
    //                 mEmitter->emitterVelocity(),
    //                 CUDA_MULTISPH_YAN16_PARAMS.rest_mass);
    //             else
    //                 mEmitter->setEmitterStatus(false);
    //             // printf("fluid particle number=%zd, max=%zd \n",
    //             mFluids->size(), mFluids->maxSize()); mEmitterCounter++;
    //         }
    //     }
    // }

    cudaDeviceSynchronize();
    KIRI_CUKERNAL();
  }

  void CudaMultiSphYan16System::onTransferGPUData2VBO(float4 *cudaPositionVBO,
                                                      float4 *cudaColorVBO)
  {
    TransferGPUData2VBO_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        cudaPositionVBO, cudaColorVBO, mFluids->posPtr(), mFluids->colorPtr(),
        CUDA_MULTISPH_YAN16_PARAMS.particle_radius, mFluids->size(), 0);

    cudaDeviceSynchronize();
    KIRI_CUKERNAL();
  }

  void CudaMultiSphYan16System::computeBoundaryVolume()
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

  void CudaMultiSphYan16System::initMultiSphSystem()
  {
    auto initCudaGridSize = CuCeilDiv(mFluids->size(), KIRI_CUBLOCKSIZE);
    InitMultiSphYan16Solver_CUDA<<<initCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        mFluids->GetYan16PhasePtr(), mFluids->phaseLabelPtr(),
        mFluids->GetPhaseTypePtr(), mFluids->GetRestPhaseType0Ptr(),
        mFluids->size(), CUDA_MULTISPH_YAN16_PARAMS.phase_num);

    cudaDeviceSynchronize();
    KIRI_CUKERNAL();
  }

} // namespace KIRI
