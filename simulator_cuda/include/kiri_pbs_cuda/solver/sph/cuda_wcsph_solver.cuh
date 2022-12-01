/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 19:05:14
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-15 18:04:55
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sph\cuda_wcsph_solver.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_WCSPH_SOLVER_CUH_
#define _CUDA_WCSPH_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver.cuh>

namespace KIRI
{
    class CudaWCSphSolver final : public CudaSphSolver
    {
    public:
        virtual void updateSolver(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            float timeIntervalInSeconds,
            CudaSphParams params,
            CudaBoundaryParams bparams) override;

        explicit CudaWCSphSolver(
            const size_t num,
            const bool cubicKernel = false,
            const float negativeScale = 0.f,
            const float timeStepLimitScale = 1.f,
            const float speedOfSound = 100.f)
            : CudaSphSolver(num),
              bCubicKernel(cubicKernel),
              mNegativeScale(negativeScale),
              mTimeStepLimitScale(timeStepLimitScale),
              mSpeedOfSound(speedOfSound)
        {
        }

        virtual ~CudaWCSphSolver() noexcept {}

        float speedOfSound() const;
        float timeStepLimitScale() const;

        void setTimeStepLimitScale(float newScale);
        void setSpeedOfSound(float newSpeedOfSound);

    private:
        float mNegativeScale, mTimeStepLimitScale, mSpeedOfSound;
        bool bCubicKernel;
        const float mTimeStepLimitBySpeedFactor = 0.4f;
        const float mTimeStepLimitByForceFactor = 0.25f;

        void computeSubTimeStepsByCFL(
            CudaSphParticlesPtr &fluids,
            const float restMass,
            const float kernelRadius,
            float timeIntervalInSeconds);

        virtual void computeDensity(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const float rho0,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize) override;

        virtual void computeNablaTerm(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize,
            const float rho0,
            const float stiff) override;

        virtual void computeViscosityTerm(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float rho0,
            const float visc,
            const float bnu,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize) override;

        virtual void computeArtificialViscosityTerm(
            CudaSphParticlesPtr &fluids,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float rho0,
            const float nu,
            const float bnu,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize) override;
    };

    typedef SharedPtr<CudaWCSphSolver> CudaWCSphSolverPtr;
} // namespace KIRI

#endif /* _CUDA_WCSPH_SOLVER_CUH_ */