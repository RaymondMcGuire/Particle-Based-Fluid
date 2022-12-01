/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-25 12:33:07
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-07-26 16:09:09
 * @FilePath:
 * \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\sph\cuda_dfsph_solver.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_pbs_cuda/solver/sph/cuda_dfsph_solver.cuh>

namespace KIRI
{
  void CudaDFSphSolver::updateSolver(CudaSphParticlesPtr &fluids,
                                     CudaBoundaryParticlesPtr &boundaries,
                                     const CudaArray<size_t> &cellStart,
                                     const CudaArray<size_t> &boundaryCellStart,
                                     float timeIntervalInSeconds,
                                     CudaSphParams params,
                                     CudaBoundaryParams bparams)
  {

    mNumOfSubTimeSteps = static_cast<size_t>(timeIntervalInSeconds / mDt);

    extraForces(fluids, params.gravity);

    computeDensity(fluids, boundaries, params.rest_density, cellStart,
                   boundaryCellStart, bparams.lowest_point, bparams.kernel_radius,
                   bparams.grid_size);

    computeAlpha(fluids, boundaries, params.rest_density, cellStart,
                 boundaryCellStart, bparams.lowest_point, bparams.kernel_radius,
                 bparams.grid_size);

    divergenceSolver(fluids, boundaries, params.rest_density, cellStart,
                     boundaryCellStart, bparams.lowest_point,
                     bparams.kernel_radius, bparams.grid_size);

    // if (params.sta_akinci13)
    // {
    //   computeAkinci13Normal(fluids, cellStart, bparams.lowest_point,
    //                         bparams.kernel_radius, bparams.grid_size);

    //   computeAkinci13Term(fluids, boundaries, cellStart, boundaryCellStart,
    //                       params.rest_density, params.a_beta, params.st_gamma,
    //                       bparams.lowest_point, bparams.kernel_radius,
    //                       bparams.grid_size);
    // }

    // if (params.atf_visc)
    computeArtificialViscosityTerm(fluids, boundaries, cellStart,
                                   boundaryCellStart, params.rest_density,
                                   params.nu, params.bnu, bparams.lowest_point,
                                   bparams.kernel_radius, bparams.grid_size);
    // else
    // computeViscosityTerm(fluids, boundaries, cellStart, boundaryCellStart,
    //                      params.rest_density, params.visc, params.bnu,
    //                      bparams.lowest_point, bparams.kernel_radius,
    //                      bparams.grid_size);

    // computeTimeStepsByCFL(fluids, params.particle_radius, timeIntervalInSeconds);

    velAdvect(fluids);

    pressureSolver(fluids, boundaries, params.rest_density, cellStart,
                   boundaryCellStart, bparams.lowest_point, bparams.kernel_radius,
                   bparams.grid_size);

    advect(fluids, mDt, bparams.lowest_point, bparams.highest_point,
           params.particle_radius);
  }

} // namespace KIRI