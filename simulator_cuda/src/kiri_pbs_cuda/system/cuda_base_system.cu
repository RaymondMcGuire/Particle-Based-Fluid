/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 19:05:14
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 15:11:27
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\system\cuda_base_system.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
// clang-format off
#include <kiri_pbs_cuda/system/cuda_base_system.cuh>

#include <glad/glad.h>
#include <cuda_gl_interop.h>

// clang-format on
namespace KIRI
{
  CudaBaseSystem::CudaBaseSystem(
      CudaBaseSolverPtr &solver,
      CudaGNSearcherPtr &searcher,
      CudaBoundaryParticlesPtr &boundary_particles,
      CudaGNBoundarySearcherPtr &boundarySearcher,
      const size_t maxNumOfParticles,
      const bool adaptiveSubTimeStep)
      : mSolver(std::move(solver)),
        mSearcher(std::move(searcher)),
        mBoundaries(std::move(boundary_particles)),
        mBoundarySearcher(std::move(boundarySearcher)),
        bAdaptiveSubTimeStep(adaptiveSubTimeStep)
  {
    KIRI_CUCALL(cudaMalloc((void **)&mCudaPositionVBO,
                           sizeof(float4) * maxNumOfParticles));
    KIRI_CUCALL(
        cudaMalloc((void **)&mCudaColorVBO, sizeof(float4) * maxNumOfParticles));

    // init position vbo
    size_t buf_size = maxNumOfParticles * sizeof(float4);
    glGenBuffers(1, &mPositionsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, mPositionsVBO);
    glBufferData(GL_ARRAY_BUFFER, buf_size, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // init color vbo
    size_t color_buf_size = maxNumOfParticles * sizeof(float4);
    glGenBuffers(1, &mColorsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, mColorsVBO);
    glBufferData(GL_ARRAY_BUFFER, color_buf_size, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // build boundary searcher
    mBoundarySearcher->buildGNSearcher(mBoundaries);
  }

  void CudaBaseSystem::updateWorldSize(const float3 lowestPoint, const float3 highestPoint)
  {

    CUDA_BOUNDARY_PARAMS.lowest_point = lowestPoint;
    CUDA_BOUNDARY_PARAMS.highest_point = highestPoint;
    CUDA_BOUNDARY_PARAMS.world_size = highestPoint - lowestPoint;
    CUDA_BOUNDARY_PARAMS.world_center = (highestPoint + lowestPoint) / 2.f;

    CUDA_BOUNDARY_PARAMS.grid_size = make_int3(
        (CUDA_BOUNDARY_PARAMS.highest_point - CUDA_BOUNDARY_PARAMS.lowest_point) /
        CUDA_BOUNDARY_PARAMS.kernel_radius);

    mSearcher->updateWorldSize(lowestPoint, highestPoint);
    mBoundarySearcher->updateWorldSize(lowestPoint, highestPoint);
  }

  void CudaBaseSystem::updateSystemForVBO(float renderInterval)
  {
    KIRI_CUCALL(cudaGraphicsGLRegisterBuffer(&mCudaGraphPosVBORes, mPositionsVBO,
                                             cudaGraphicsMapFlagsNone));
    KIRI_CUCALL(cudaGraphicsGLRegisterBuffer(&mCudaGraphColorVBORes, mColorsVBO,
                                             cudaGraphicsMapFlagsNone));

    size_t num_bytes = 0;
    KIRI_CUCALL(cudaGraphicsMapResources(1, &mCudaGraphPosVBORes, 0));
    KIRI_CUCALL(cudaGraphicsResourceGetMappedPointer(
        (void **)&mCudaPositionVBO, &num_bytes, mCudaGraphPosVBORes));

    size_t color_num_bytes = 0;
    KIRI_CUCALL(cudaGraphicsMapResources(1, &mCudaGraphColorVBORes, 0));
    KIRI_CUCALL(cudaGraphicsResourceGetMappedPointer(
        (void **)&mCudaColorVBO, &color_num_bytes, mCudaGraphColorVBORes));

    if (bAdaptiveSubTimeStep)
    {
      float remaining_time = renderInterval;
      while (remaining_time > KIRI_EPSILON)
      {
        updateSystem(remaining_time);
        remaining_time -=
            remaining_time / static_cast<float>(mSolver->subTimeStepsNum());
      }
    }
    else
    {
      size_t sub_timesteps_num = mSolver->subTimeStepsNum();
      for (size_t i = 0; i < sub_timesteps_num; i++)
        updateSystem(renderInterval);
    }

    onTransferGPUData2VBO(mCudaPositionVBO, mCudaColorVBO);

    KIRI_CUCALL(cudaGraphicsUnmapResources(1, &mCudaGraphPosVBORes, 0));
    KIRI_CUCALL(cudaGraphicsUnregisterResource(mCudaGraphPosVBORes));

    KIRI_CUCALL(cudaGraphicsUnmapResources(1, &mCudaGraphColorVBORes, 0));
    KIRI_CUCALL(cudaGraphicsUnregisterResource(mCudaGraphColorVBORes));
  }

  float CudaBaseSystem::updateSystem(float renderInterval)
  {
    cudaEvent_t start, stop;
    KIRI_CUCALL(cudaEventCreate(&start));
    KIRI_CUCALL(cudaEventCreate(&stop));
    KIRI_CUCALL(cudaEventRecord(start, 0));

    try
    {
      onUpdateSolver(renderInterval);
    }
    catch (const char *s)
    {
      std::cout << s << "\n";
    }
    catch (...)
    {
      std::cout << "Unknown Exception at " << __FILE__ << ": line " << __LINE__
                << "\n";
    }

    float milliseconds;
    KIRI_CUCALL(cudaEventRecord(stop, 0));
    KIRI_CUCALL(cudaEventSynchronize(stop));
    KIRI_CUCALL(cudaEventElapsedTime(&milliseconds, start, stop));
    KIRI_CUCALL(cudaEventDestroy(start));
    KIRI_CUCALL(cudaEventDestroy(stop));
    return milliseconds;
  }
} // namespace KIRI
