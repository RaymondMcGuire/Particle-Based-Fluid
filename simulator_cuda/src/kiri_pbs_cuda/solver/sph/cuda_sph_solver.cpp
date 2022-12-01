/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 19:05:14
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-07 10:35:58
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\sph\cuda_sph_solver.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver.cuh>

namespace KIRI
{
  void CudaSphSolver::updateSolver(CudaSphParticlesPtr &fluids,
                                   CudaBoundaryParticlesPtr &boundaries,
                                   const CudaArray<size_t> &cellStart,
                                   const CudaArray<size_t> &boundaryCellStart,
                                   float timeIntervalInSeconds,
                                   CudaSphParams params,
                                   CudaBoundaryParams bparams)
  {
    mNumOfSubTimeSteps = static_cast<size_t>(timeIntervalInSeconds / params.dt);
    // printf("mNumOfSubTimeSteps=%d \n", mNumOfSubTimeSteps);

    extraForces(fluids, params.gravity);

    computeDensity(fluids, boundaries, params.rest_density, cellStart,
                   boundaryCellStart, bparams.lowest_point, bparams.kernel_radius,
                   bparams.grid_size);

    computeNablaTerm(fluids, boundaries, cellStart, boundaryCellStart,
                     bparams.lowest_point, bparams.kernel_radius,
                     bparams.grid_size, params.rest_density, params.stiff);

    if (params.sta_akinci13)
    {
      computeAkinci13Normal(fluids, cellStart, bparams.lowest_point,
                            bparams.kernel_radius, bparams.grid_size);

      computeAkinci13Term(fluids, boundaries, cellStart, boundaryCellStart,
                          params.rest_density, params.a_beta, params.st_gamma,
                          bparams.lowest_point, bparams.kernel_radius,
                          bparams.grid_size);
    }

    if (params.atf_visc)
      computeArtificialViscosityTerm(fluids, boundaries, cellStart,
                                     boundaryCellStart, params.rest_density,
                                     params.nu, params.bnu, bparams.lowest_point,
                                     bparams.kernel_radius, bparams.grid_size);
    else
      computeViscosityTerm(fluids, boundaries, cellStart, boundaryCellStart,
                           params.rest_density, params.visc, params.bnu,
                           bparams.lowest_point, bparams.kernel_radius,
                           bparams.grid_size);

    advect(fluids, params.dt, bparams.lowest_point, bparams.highest_point,
           params.particle_radius);
  }

} // namespace KIRI