/***
 * @Author: Xu.WANG
 * @Date: 2021-02-03 16:35:31
 * @LastEditTime: 2021-04-22 22:15:28
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\multisph\cuda_multiwcsph_ren14_solver.cpp
 */

#include <kiri_pbs_cuda/solver/multisph/cuda_multiwcsph_ren14_solver.cuh>

namespace KIRI
{
  void CudaMultiWCSphRen14Solver::updateSolver(
      CudaMultiSphRen14ParticlesPtr &particles,
      CudaBoundaryParticlesPtr &boundaries, const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, float timeIntervalInSeconds,
      CudaMultiSphRen14Params params, CudaBoundaryParams bparams)
  {
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

    correctVolumeFraction(particles, params.phase_num, params.miscible, params.stiff,
                          params.dt);

    computeMultiSphAcc(particles, boundaries, params.phase_num, params.gravity,
                       params.sound_speed, params.bnu, cellStart,
                       boundaryCellStart, bparams.lowest_point,
                       bparams.kernel_radius, bparams.grid_size);

    computeNextTimeStepData(particles, params.phase_num, params.miscible);

    computeSubTimeStepsByCFL(particles, params.phase_num, params.mass0, params.dt,
                             bparams.kernel_radius, timeIntervalInSeconds);

    advect(particles, params.dt, bparams.lowest_point, bparams.highest_point,
           params.particle_radius);
  }

} // namespace KIRI