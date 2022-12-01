/*
 * @Author: Xu.WANG
 * @Date: 2020-07-04 14:48:23
 * @LastEditTime: 2021-04-23 10:22:52
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\multisph\cuda_multiwcsph_yan16_solver_gpu.cuh
 */

#ifndef _CUDA_MULTIWCSPH_YAN16_SOLVER_GPU_CUH_
#define _CUDA_MULTIWCSPH_YAN16_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/data/cuda_multisph_params.h>
#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver_common_gpu.cuh>

namespace KIRI
{
    __global__ void ComputeMultiWCSphYanPressure_CUDA(
        Yan16PhaseData *phaseDataBlock1,
        const float *avgDensity,
        const size_t num,
        const size_t phaseNum,
        const bool miscible,
        const float stiff,
        const float negativeScale)
    {
        const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num)
            return;

        float rest_mix_pressure = stiff * (powf((avgDensity[i] / phaseDataBlock1[i].rest_mix_density), 7.f) - 1.f);

        if (rest_mix_pressure < 0.f)
            rest_mix_pressure *= negativeScale;

        if (miscible)
            for (size_t k = 0; k < phaseNum; k++)
                phaseDataBlock1[i].phase_pressure[k] = phaseDataBlock1[i].volume_fractions[k] * rest_mix_pressure;
        else
            for (size_t k = 0; k < phaseNum; k++)
                phaseDataBlock1[i].phase_pressure[k] = rest_mix_pressure;

        phaseDataBlock1[i].rest_mix_pressure = rest_mix_pressure;

        return;
    }

} // namespace KIRI

#endif /* _CUDA_MULTIWCSPH_YAN16_SOLVER_GPU_CUH_ */