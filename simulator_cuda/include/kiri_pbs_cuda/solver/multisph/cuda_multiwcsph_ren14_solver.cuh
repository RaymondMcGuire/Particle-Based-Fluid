/*
 * @Author: Xu.WANG
 * @Date: 2021-02-01 14:31:30
 * @LastEditTime: 2021-04-22 22:13:56
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\multisph\cuda_multiwcsph_ren14_solver.cuh
 */

#ifndef _CUDA_MULTIWCSPH_REN14_SOLVER_CUH_
#define _CUDA_MULTIWCSPH_REN14_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/multisph/cuda_multisph_ren14_solver.cuh>

namespace KIRI
{
    class CudaMultiWCSphRen14Solver final : public CudaMultiSphRen14Solver
    {
    public:
        explicit CudaMultiWCSphRen14Solver(
            const size_t num,
            const float negativeScale = 0.f,
            const float timeStepLimitScale = 3.f,
            const float speedOfSound = 100.f)
            : CudaMultiSphRen14Solver(num),
              mNegativeScale(negativeScale),
              mTimeStepLimitScale(timeStepLimitScale),
              mSpeedOfSound(speedOfSound)
        {
        }

        virtual ~CudaMultiWCSphRen14Solver() noexcept {}

        virtual void updateSolver(
            CudaMultiSphRen14ParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            float timeIntervalInSeconds,
            CudaMultiSphRen14Params params,
            CudaBoundaryParams bparams) override;

    protected:
        virtual void computeMixPressure(
            CudaMultiSphRen14ParticlesPtr &particles,
            const size_t phaseNum,
            const bool miscible,
            const float stiff) override;

    private:
        float mNegativeScale, mTimeStepLimitScale, mSpeedOfSound;
        const float mTimeStepLimitBySpeedFactor = 0.4f;
        const float mTimeStepLimitByForceFactor = 0.25f;

        void computeSubTimeStepsByCFL(
            CudaMultiSphRen14ParticlesPtr &particles,
            const size_t phaseNum,
            const float sphMass[MULTISPH_MAX_PHASE_NUM],
            const float dt,
            const float kernelRadius,
            float renderInterval);
    };

    typedef SharedPtr<CudaMultiWCSphRen14Solver> CudaMultiWCSphRen14SolverPtr;
} // namespace KIRI

#endif /* _CUDA_MULTIWCSPH_REN14_SOLVER_CUH_ */