/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-11 11:44:38
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-15 18:12:15
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\sph\cuda_multi_sph_solver.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_pbs_cuda/solver/sph/cuda_multi_sph_solver.cuh>

namespace KIRI
{
  void CudaMultiSphSolver::updateSolver(
      CudaSphParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, float timeIntervalInSeconds,
      CudaSphParams params, CudaBoundaryParams bparams)
  {
    mNumOfSubTimeSteps = static_cast<size_t>(timeIntervalInSeconds / params.dt);

    extraForces(fluids, params.gravity);

    computeMRDensity(fluids, boundaries, params.rest_density, cellStart,
                     boundaryCellStart, bparams.lowest_point,
                     params.kernel_radius, params.grid_size);

    computeMRNablaTerm(fluids, boundaries, cellStart, boundaryCellStart,
                       bparams.lowest_point, params.kernel_radius,
                       params.grid_size, params.rest_density, params.stiff);

    computeMRArtificialViscosityTerm(fluids, boundaries, cellStart,
                                     boundaryCellStart, params.rest_density,
                                     params.nu, params.bnu, bparams.lowest_point,
                                     params.kernel_radius, params.grid_size);

    advectMRSph(fluids, params.dt, bparams.lowest_point, bparams.highest_point);
  }

} // namespace KIRI