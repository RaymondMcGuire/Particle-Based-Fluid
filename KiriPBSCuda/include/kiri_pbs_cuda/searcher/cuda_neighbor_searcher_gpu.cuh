/*
 * @Author: Xu.WANG
 * @Date: 2020-10-18 02:13:36
 * @LastEditTime: 2021-02-06 23:20:41
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\searcher\cuda_neighbor_searcher_gpu.cuh
 */
#ifndef _CUDA_NEIGHBOR_SEARCHER_GPU_CUH_
#define _CUDA_NEIGHBOR_SEARCHER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>
namespace KIRI
{

    __global__ void CountingInCell_CUDA(uint *cellStart, uint *particle2cell, const uint num)
    {
        const uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num)
            return;
        atomicAdd(&cellStart[particle2cell[i]], 1);
        return;
    }
} // namespace KIRI
#endif /* _CUDA_NEIGHBOR_SEARCHER_GPU_CUH_ */