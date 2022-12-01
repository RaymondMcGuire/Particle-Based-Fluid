/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-24 14:09:09
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-24 15:15:48
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\sph\cuda_pbf_solver.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_pbs_cuda/solver/sph/cuda_pbf_solver.cuh>

namespace KIRI {
void CudaPBFSolver::updateSolver(CudaSphParticlesPtr &fluids,
                                 CudaBoundaryParticlesPtr &boundaries,
                                 const CudaArray<size_t> &cellStart,
                                 const CudaArray<size_t> &boundaryCellStart,
                                 float timeIntervalInSeconds,
                                 CudaSphParams params,
                                 CudaBoundaryParams bparams) {
  mNumOfSubTimeSteps = static_cast<size_t>(timeIntervalInSeconds / params.dt);
  auto particles = std::dynamic_pointer_cast<CudaPBFParticles>(fluids);

  extraForces(fluids, params.gravity);

  if (mIncompressiable) {
    size_t iterations = 0;
    bool flag = false;
    const float eta = MAX_DENSITY_ERROR * 0.01f * params.rest_density;

    while ((!flag || (iterations < MIN_ITERATION)) &&
           (iterations < MAX_ITERATION)) {
      flag = true;

      constaintProjection(particles, boundaries, params.rest_density, params.dt,
                          cellStart, boundaryCellStart, bparams.lowest_point,
                          bparams.kernel_radius, bparams.grid_size);

      flag = flag && (mAvgDensityError <= eta);

      iterations++;
    }

    printf("iterations=%zd, mAvgDensityError=%.3f \n", iterations,
           mAvgDensityError);
  } else {
    for (auto iter = 0; iter < MAX_REALTIME_ITERATION; iter++)
      constaintProjection(particles, boundaries, params.rest_density, params.dt,
                          cellStart, boundaryCellStart, bparams.lowest_point,
                          bparams.kernel_radius, bparams.grid_size);
  }

  particles->updateVelFirstOrder(params.dt);

  computeViscosityXSPH(particles, boundaries, params.rest_density, mVisc,
                       mBoundaryVisc, params.dt, cellStart, boundaryCellStart,
                       bparams.lowest_point, bparams.kernel_radius,
                       bparams.grid_size);

  if (mIncompressiable)
    computeVorticityConfinement(particles, mVorticityCoeff, cellStart,
                                bparams.lowest_point, bparams.kernel_radius,
                                bparams.grid_size);

  advect(fluids, params.dt, bparams.lowest_point, bparams.highest_point,
         params.particle_radius);
}

} // namespace KIRI