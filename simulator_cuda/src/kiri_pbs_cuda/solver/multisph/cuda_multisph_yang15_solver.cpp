/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-08-24 14:09:09
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-24 14:53:24
 * @FilePath:
 * \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\multisph\cuda_multisph_yang15_solver.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_pbs_cuda/solver/multisph/cuda_multisph_yang15_solver.cuh>

namespace KIRI {
void CudaMultiSphYang15Solver::updateSolver(
    CudaMultiSphYang15ParticlesPtr &particles,
    CudaBoundaryParticlesPtr &boundaries, const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, float timeIntervalInSeconds,
    CudaMultiSphYang15Params params, CudaBoundaryParams bparams) {
  mNumOfSubTimeSteps = static_cast<size_t>(timeIntervalInSeconds / params.dt);

  extraForces(particles, params.gravity);

  computeAggregateDensity(particles, boundaries, params.phase_num, cellStart,
                          boundaryCellStart, bparams.lowest_point,
                          bparams.kernel_radius, bparams.grid_size);

  computeNSCHModel(particles, params.phase_num, params.eta, params.mobilities,
                   params.alpha, params.s1, params.s2, params.epsilon,
                   cellStart, bparams.lowest_point, bparams.kernel_radius,
                   bparams.grid_size);

  for (size_t i = 0; i < 4; i++) {

    constaintProjection(particles, boundaries, params.dt, cellStart,
                        boundaryCellStart, bparams.lowest_point,
                        bparams.kernel_radius, bparams.grid_size);
  }
  // velocity update
  particles->updateVelFirstOrder(params.dt);

  computeViscosityXSPH(particles, boundaries, params.visc, params.boundary_visc,
                       params.dt, cellStart, boundaryCellStart,
                       bparams.lowest_point, bparams.kernel_radius,
                       bparams.grid_size);

  computeSurfaceTension(particles, params.sigma, params.eta, params.epsilon,
                        params.dt, params.phase_num, cellStart,
                        bparams.lowest_point, bparams.kernel_radius,
                        bparams.grid_size);

  computeNextTimeStepData(particles, params.phase_num);

  advect(particles, params.dt, bparams.lowest_point, bparams.highest_point,
         params.particle_radius);
}

} // namespace KIRI