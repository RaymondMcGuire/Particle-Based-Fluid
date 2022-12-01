/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-05-18 21:25:47
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-15 17:46:46
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sph\cuda_sph_solver_gpu.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_SPH_SOLVER_GPU_CUH_
#define _CUDA_SPH_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver_common_gpu.cuh>

namespace KIRI
{

  template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
            typename GradientFunc>
  __global__ void _ComputeNablaTerm_CUDA(
      float3 *pos, float3 *acc, float *mass, float *density, float *pressure,
      const float rho0, const size_t num, size_t *cellStart, float3 *bPos,
      float *bVolume, size_t *bCellStart, const int3 gridSize, Pos2GridXYZ p2xyz,
      GridXYZ2GridHash xyz2hash, GradientFunc nablaW)
  {
    const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
      return;

    auto a = make_float3(0.f);
    int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
    for (int m = 0; m < 27; ++m)
    {
      int3 cur_grid_xyz =
          grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
      const size_t hash_idx =
          xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
      if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
        continue;

      _ComputeFluidPressure(&a, i, pos, mass, density, pressure,
                            cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
      _computeBoundaryPressure(&a, pos[i], density[i], pressure[i], bPos, bVolume,
                               rho0, bCellStart[hash_idx],
                               bCellStart[hash_idx + 1], nablaW);
    }
    if (a.x != a.x || a.y != a.y || a.z != a.z)
    {
      printf("_ComputeMuller03Viscosity acc nan!! a=%.3f,%.3f,%.3f \n",
             KIRI_EXPAND_FLOAT3(a));
    }
    acc[i] += a;
    return;
  }

} // namespace KIRI

#endif /* _CUDA_SPH_SOLVER_GPU_CUH_ */