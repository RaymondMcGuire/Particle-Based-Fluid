/***
 * @Author: Xu.WANG
 * @Date: 2021-02-03 16:35:31
 * @LastEditTime: 2021-03-14 23:03:35
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\sph\cuda_wcsph_solver.cpp
 */

#include <kiri_pbs_cuda/solver/sph/cuda_wcsph_solver.cuh>

namespace KIRI
{
  float CudaWCSphSolver::speedOfSound() const { return mSpeedOfSound; }

  void CudaWCSphSolver::setSpeedOfSound(float newSpeedOfSound)
  {
    mSpeedOfSound = std::max(newSpeedOfSound, KIRI_EPSILON);
  }

  float CudaWCSphSolver::timeStepLimitScale() const
  {
    return mTimeStepLimitScale;
  }

  void CudaWCSphSolver::setTimeStepLimitScale(float newScale)
  {
    mTimeStepLimitScale = std::max(newScale, 0.f);
  }

  void CudaWCSphSolver::updateSolver(CudaSphParticlesPtr &fluids,
                                     CudaBoundaryParticlesPtr &boundaries,
                                     const CudaArray<size_t> &cellStart,
                                     const CudaArray<size_t> &boundaryCellStart,
                                     float timeIntervalInSeconds,
                                     CudaSphParams params,
                                     CudaBoundaryParams bparams)
  {
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

    computeSubTimeStepsByCFL(fluids, params.rest_mass, bparams.kernel_radius,
                             timeIntervalInSeconds);

    advect(fluids, timeIntervalInSeconds / static_cast<float>(mNumOfSubTimeSteps),
           bparams.lowest_point, bparams.highest_point, params.particle_radius);
  }

} // namespace KIRI