/*
 * @Author: Xu.WANG
 * @Date: 2021-02-04 12:36:10
 * @LastEditTime: 2021-02-09 12:46:20
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_particles.cuh
 */

#ifndef _CUDA_PARTICLES_CUH_
#define _CUDA_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/data/cuda_array.cuh>

namespace KIRI
{
    class CudaParticles
    {
    public:
        explicit CudaParticles(const Vec_Float3 &p) : mPos(p.size())
        {
            KIRI_CUCALL(cudaMemcpy(mPos.Data(), &p[0], sizeof(float3) * p.size(), cudaMemcpyHostToDevice));
        }

        CudaParticles(const CudaParticles &) = delete;
        CudaParticles &operator=(const CudaParticles &) = delete;

        uint Size() const { return mPos.Length(); }
        float3 *GetPosPtr() const { return mPos.Data(); }
        virtual ~CudaParticles() noexcept {}

    protected:
        CudaArray<float3> mPos;
    };

    typedef SharedPtr<CudaParticles> CudaParticlesPtr;
} // namespace KIRI

#endif