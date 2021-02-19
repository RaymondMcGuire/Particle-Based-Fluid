/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 22:59:48
 * @LastEditTime: 2021-02-13 23:29:39
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\system\cuda_sph_system.cu
 */

#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
#include <kiri_pbs_cuda/system/cuda_sph_system.cuh>
#include <kiri_pbs_cuda/system/cuda_sph_system_gpu.cuh>

#include <glad/glad.h>
#include <cuda_gl_interop.h>
namespace KIRI
{

    CudaSphSystem::CudaSphSystem(
        CudaSphParticlesPtr &fluidParticles,
        CudaBoundaryParticlesPtr &boundaryParticles,
        CudaBaseSolverPtr &solver,
        CudaGNSearcherPtr &searcher,
        CudaGNBoundarySearcherPtr &boundarySearcher,
        bool openGL)
        : mFluids(std::move(fluidParticles)),
          mBoundaries(std::move(boundaryParticles)),
          mSolver(std::move(solver)),
          mSearcher(std::move(searcher)),
          mBoundarySearcher(std::move(boundarySearcher)),
          bOpenGL(openGL),
          mCudaGridSize(CuCeilDiv(mFluids->Size(), KIRI_CUBLOCKSIZE))
    {

        uint maxNumOfParticles = mFluids->Size();

        KIRI_CUCALL(cudaMalloc((void **)&pptr, sizeof(float4) * maxNumOfParticles));
        KIRI_CUCALL(cudaMalloc((void **)&cptr, sizeof(float4) * maxNumOfParticles));

        // init position vbo
        uint bufSize = maxNumOfParticles * sizeof(float4);
        glGenBuffers(1, &mPositionsVBO);
        glBindBuffer(GL_ARRAY_BUFFER, mPositionsVBO);
        glBufferData(GL_ARRAY_BUFFER, bufSize, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // init color vbo
        uint colorBufSize = maxNumOfParticles * sizeof(float4);
        glGenBuffers(1, &mColorsVBO);
        glBindBuffer(GL_ARRAY_BUFFER, mColorsVBO);
        glBufferData(GL_ARRAY_BUFFER, colorBufSize, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // build boundary searcher
        mBoundarySearcher->BuildGNSearcher(mBoundaries);

        // compute boundary volume(Akinci2012)
        ComputeBoundaryVolume();

        // init fluid system
        thrust::fill(thrust::device, mFluids->GetMassPtr(), mFluids->GetMassPtr() + mFluids->Size(), CUDA_SPH_PARAMS.rest_mass);

        if (bOpenGL)
            UpdateSystemForVBO();
        else
            UpdateSystem();
    }

    void CudaSphSystem::UpdateSystemForVBO()
    {
        KIRI_CUCALL(cudaGraphicsGLRegisterBuffer(&mCudaGraphPosVBORes, mPositionsVBO,
                                                 cudaGraphicsMapFlagsNone));
        KIRI_CUCALL(cudaGraphicsGLRegisterBuffer(&mCudaGraphColorVBORes, mColorsVBO,
                                                 cudaGraphicsMapFlagsNone));

        size_t numBytes = 0;
        KIRI_CUCALL(cudaGraphicsMapResources(1, &mCudaGraphPosVBORes, 0));
        KIRI_CUCALL(cudaGraphicsResourceGetMappedPointer(
            (void **)&pptr, &numBytes, mCudaGraphPosVBORes));

        size_t colorNumBytes = 0;
        KIRI_CUCALL(cudaGraphicsMapResources(1, &mCudaGraphColorVBORes, 0));
        KIRI_CUCALL(cudaGraphicsResourceGetMappedPointer(
            (void **)&cptr, &colorNumBytes, mCudaGraphColorVBORes));

        UpdateSystem();

        CopyGPUData2VBO(pptr, cptr, mFluids);

        KIRI_CUCALL(cudaGraphicsUnmapResources(1, &mCudaGraphPosVBORes, 0));
        KIRI_CUCALL(cudaGraphicsUnregisterResource(mCudaGraphPosVBORes));

        KIRI_CUCALL(cudaGraphicsUnmapResources(1, &mCudaGraphColorVBORes, 0));
        KIRI_CUCALL(cudaGraphicsUnregisterResource(mCudaGraphColorVBORes));
    }

    void CudaSphSystem::CopyGPUData2VBO(float4 *pos, float4 *col, const CudaSphParticlesPtr &fluids)
    {

        CopyGPUData2VBO_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(pos, col, fluids->GetPosPtr(), fluids->GetColPtr(), fluids->Size(), CUDA_SPH_PARAMS.particle_radius);

        KIRI_CUKERNAL();
    }

    void CudaSphSystem::ComputeBoundaryVolume()
    {
        auto mCudaBoundaryGridSize = CuCeilDiv(mBoundaries->Size(), KIRI_CUBLOCKSIZE);

        ComputeBoundaryVolume_CUDA<<<mCudaBoundaryGridSize, KIRI_CUBLOCKSIZE>>>(
            mBoundaries->GetPosPtr(),
            mBoundaries->GetVolumePtr(),
            mBoundaries->Size(),
            mBoundarySearcher->GetCellStartPtr(),
            mBoundarySearcher->GetGridSize(),
            ThrustHelper::Pos2GridXYZ<float3>(mBoundarySearcher->GetLowestPoint(), mBoundarySearcher->GetCellSize(), mBoundarySearcher->GetGridSize()),
            ThrustHelper::GridXYZ2GridHash(mBoundarySearcher->GetGridSize()),
            Poly6Kernel(mBoundarySearcher->GetCellSize()));
        KIRI_CUKERNAL();
    }

    float CudaSphSystem::UpdateSystem()
    {
        cudaEvent_t start, stop;
        KIRI_CUCALL(cudaEventCreate(&start));
        KIRI_CUCALL(cudaEventCreate(&stop));
        KIRI_CUCALL(cudaEventRecord(start, 0));

        mSearcher->BuildGNSearcher(mFluids);
        try
        {
            mSolver->UpdateSolver(
                mFluids,
                mBoundaries,
                mSearcher->GetCellStart(),
                mBoundarySearcher->GetCellStart(),
                CUDA_SPH_PARAMS,
                CUDA_BOUNDARY_PARAMS);
            cudaDeviceSynchronize();
            KIRI_CUKERNAL();
        }
        catch (const char *s)
        {
            std::cout << s << "\n";
        }
        catch (...)
        {
            std::cout << "Unknown Exception at " << __FILE__ << ": line " << __LINE__ << "\n";
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
