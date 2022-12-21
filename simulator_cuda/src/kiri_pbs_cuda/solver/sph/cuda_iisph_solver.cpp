/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-12-01 23:00:37
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-12-21 17:08:02
 * @FilePath:
 * \Particle-Based-Fluid-Toolkit\simulator_cuda\src\kiri_pbs_cuda\solver\sph\cuda_iisph_solver.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_pbs_cuda/solver/sph/cuda_iisph_solver.cuh>

namespace KIRI {
void CudaIISphSolver::updateSolver(CudaSphParticlesPtr &fluids,
                                   CudaBoundaryParticlesPtr &boundaries,
                                   const CudaArray<size_t> &cellStart,
                                   const CudaArray<size_t> &boundaryCellStart,
                                   float timeIntervalInSeconds,
                                   CudaSphParams params,
                                   CudaBoundaryParams bparams) {

  mNumOfSubTimeSteps = static_cast<size_t>(timeIntervalInSeconds / params.dt);

  extraForces(fluids, params.gravity);

  computeDensity(fluids, boundaries, params.rest_density, cellStart,
                 boundaryCellStart, bparams.lowest_point, bparams.kernel_radius,
                 bparams.grid_size);

  if (params.sta_akinci13) {
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

  predictVelAdvect(fluids, params.dt);

  computeDiiTerm(fluids, boundaries, cellStart, boundaryCellStart,
                 params.rest_density, bparams.lowest_point,
                 bparams.kernel_radius, bparams.grid_size);

  computeAiiTerm(fluids, boundaries, cellStart, boundaryCellStart,
                 params.rest_density, params.dt, bparams.lowest_point,
                 bparams.kernel_radius, bparams.grid_size);

  pressureSolver(fluids, boundaries, params.rest_density, params.dt, cellStart,
                 boundaryCellStart, bparams.lowest_point, bparams.kernel_radius,
                 bparams.grid_size);

  computePressureAcceleration(fluids, boundaries, cellStart, boundaryCellStart,
                              params.rest_density, bparams.lowest_point,
                              bparams.kernel_radius, bparams.grid_size);

  advect(fluids, params.dt, bparams.lowest_point, bparams.highest_point,
         params.particle_radius);
}

} // namespace KIRI