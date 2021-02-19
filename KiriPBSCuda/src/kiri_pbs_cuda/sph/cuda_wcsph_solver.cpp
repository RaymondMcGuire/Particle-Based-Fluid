/*** 
 * @Author: Xu.WANG
 * @Date: 2021-02-03 16:35:31
 * @LastEditTime: 2021-02-15 14:08:53
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\sph\cuda_wcsph_solver.cpp
 */

#include <kiri_pbs_cuda/sph/cuda_wcsph_solver.cuh>

namespace KIRI
{
    void CudaWCSphSolver::UpdateSolver(
        CudaSphParticlesPtr &fluids,
        CudaBoundaryParticlesPtr &boundaries,
        const CudaArray<uint> &cellStart,
        const CudaArray<uint> &boundaryCellStart,
        CudaSphParams params,
        CudaBoundaryParams bparams)
    {
        ExtraForces(
            fluids,
            params.gravity);

        ComputeDensity(
            fluids,
            boundaries,
            params.rest_density,
            cellStart,
            boundaryCellStart,
            bparams.lowest_point,
            bparams.kernel_radius,
            bparams.grid_size);

        ComputeNablaTerm(
            fluids,
            boundaries,
            cellStart,
            boundaryCellStart,
            bparams.lowest_point,
            bparams.kernel_radius,
            bparams.grid_size,
            params.rest_density,
            params.stiff);

        if (params.atf_visc)
            ComputeArtificialViscosityTerm(
                fluids,
                boundaries,
                cellStart,
                boundaryCellStart,
                params.rest_density,
                params.nu,
                params.bnu,
                bparams.lowest_point,
                bparams.kernel_radius,
                bparams.grid_size);
        else
            ComputeViscosityTerm(
                fluids,
                boundaries,
                cellStart,
                boundaryCellStart,
                params.rest_density,
                params.visc,
                params.bnu,
                bparams.lowest_point,
                bparams.kernel_radius,
                bparams.grid_size);

        Advect(
            fluids,
            params.dt,
            bparams.lowest_point,
            bparams.highest_point,
            params.particle_radius);
    }

} // namespace KIRI