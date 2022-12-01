/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 14:46:50
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-09 17:59:45
 * @FilePath:
 * \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\multisph\cuda_multisph_ren14_solver.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_pbs_cuda/solver/multisph/cuda_multisph_ren14_solver.cuh>

namespace KIRI
{
  void CudaMultiSphRen14Solver::updateSolver(
      CudaMultiSphRen14ParticlesPtr &particles,
      CudaBoundaryParticlesPtr &boundaries, const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, float timeIntervalInSeconds,
      CudaMultiSphRen14Params params, CudaBoundaryParams bparams)
  {
    mNumOfSubTimeSteps = static_cast<size_t>(timeIntervalInSeconds / params.dt);

    computeMixDensity(particles, boundaries, params.phase_num, cellStart,
                      boundaryCellStart, bparams.lowest_point,
                      bparams.kernel_radius, bparams.grid_size);

    computeMixPressure(particles, params.phase_num, params.miscible,
                       params.stiff);

    computeGradientTerm(particles, params.phase_num, params.miscible, cellStart,
                        bparams.lowest_point, bparams.kernel_radius,
                        bparams.grid_size);

    computeDriftVelocities(particles, params.phase_num, params.miscible,
                           params.tou, params.sigma, params.gravity);

    computeDeltaVolumeFraction(particles, params.phase_num, params.miscible,
                               cellStart, bparams.lowest_point,
                               bparams.kernel_radius, bparams.grid_size);

    correctVolumeFraction(
        particles,
        params.phase_num,
        params.miscible,
        params.stiff,
        params.dt);

    // computeRestMixData(particles, params.phase_num);

    computeMultiSphAcc(particles, boundaries, params.phase_num, params.gravity,
                       params.sound_speed, params.bnu, cellStart,
                       boundaryCellStart, bparams.lowest_point,
                       bparams.kernel_radius, bparams.grid_size);

    computeNextTimeStepData(
        particles,
        params.phase_num,
        params.miscible);

    // printf("dt=%.3f\n", params.dt);

    advect(particles, params.dt, bparams.lowest_point, bparams.highest_point,
           params.particle_radius);
  }

} // namespace KIRI