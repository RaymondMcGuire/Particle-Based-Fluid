/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2021-03-17 03:21:35
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-15 18:09:34
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\cuda_solver_common_gpu.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_SOLVER_COMMON_GPU_CUH_
#define _CUDA_SOLVER_COMMON_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>
#include <curand.h>
#include <curand_kernel.h>
namespace KIRI
{

    static __global__ void _WorldBoundaryConstrain_CUDA(
        float3 *pos,
        float3 *vel,
        const size_t num,
        const float3 lowestPoint,
        const float3 highestPoint,
        const float radius)
    {
        const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num)
            return;

        float3 tmp_pos = pos[i];
        float3 tmp_vel = vel[i];

        if (tmp_pos.x > highestPoint.x - 2 * radius)
        {
            tmp_pos.x = highestPoint.x - 2 * radius;
            tmp_vel.x = fminf(tmp_vel.x, 0.f);
        }

        if (tmp_pos.x < lowestPoint.x + 2 * radius)
        {
            tmp_pos.x = lowestPoint.x + 2 * radius;
            tmp_vel.x = fmaxf(tmp_vel.x, 0.f);
        }

        if (tmp_pos.y > highestPoint.y - 2 * radius)
        {
            tmp_pos.y = highestPoint.y - 2 * radius;
            tmp_vel.y = fminf(tmp_vel.y, 0.f);
        }

        if (tmp_pos.y < lowestPoint.y + 2 * radius)
        {
            tmp_pos.y = lowestPoint.y + 2 * radius;
            tmp_vel.y = fmaxf(tmp_vel.y, 0.f);
        }

        if (tmp_pos.z > highestPoint.z - 2 * radius)
        {
            tmp_pos.z = highestPoint.z - 2 * radius;
            tmp_vel.z = fminf(tmp_vel.z, 0.f);
        }

        if (tmp_pos.z < lowestPoint.z + 2 * radius)
        {
            tmp_pos.z = lowestPoint.z + 2 * radius;
            tmp_vel.z = fmaxf(tmp_vel.z, 0.f);
        }

        pos[i] = tmp_pos;
        vel[i] = tmp_vel;

        return;
    }

    // generates a random float between 0 and 1
    static __device__ float _RndFloat_CUDA(curandState *globalState, int ind)
    {
        curandState local_state = globalState[ind];
        float val = curand_uniform(&local_state);
        globalState[ind] = local_state;
        return val;
    }

    static __global__ void _SetUpRndGen_CUDA(curandState *state, unsigned long seed)
    {
        int id = threadIdx.x;
        curand_init(seed, id, 0, &state[id]);
    }

} // namespace KIRI

#endif /* _CUDA_SOLVER_COMMON_GPU_CUH_ */