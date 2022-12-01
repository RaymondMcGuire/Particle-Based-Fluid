/*
 * @Author: Xu.WANG
 * @Date: 2021-02-01 14:31:30
 * @LastEditTime: 2021-04-23 10:24:20
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\multisph\cuda_multiwcsph_yan16_solver.cuh
 */

#ifndef _CUDA_MULTIWCSPH_YAN16_SOLVER_CUH_
#define _CUDA_MULTIWCSPH_YAN16_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/multisph/cuda_multisph_yan16_solver.cuh>

namespace KIRI
{
    class CudaMultiWCSphYan16Solver final : public CudaMultiSphYan16Solver
    {
    public:
        explicit CudaMultiWCSphYan16Solver(
            const size_t num,
            const float negativeScale = 0.f,
            const float timeStepLimitScale = 3.f,
            const float speedOfSound = 100.f)
            : CudaMultiSphYan16Solver(num),
              mNegativeScale(negativeScale),
              mTimeStepLimitScale(timeStepLimitScale),
              mSpeedOfSound(speedOfSound)
        {
        }

        virtual ~CudaMultiWCSphYan16Solver() noexcept {}

        virtual void updateSolver(
            CudaMultiSphYan16ParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            float timeIntervalInSeconds,
            CudaMultiSphYan16Params params,
            CudaBoundaryParams bparams) override;

    protected:
        virtual void computeMixPressure(
            CudaMultiSphYan16ParticlesPtr &particles,
            const size_t phaseNum,
            const bool miscible,
            const float stiff) override;

    private:
        float mNegativeScale, mTimeStepLimitScale, mSpeedOfSound;
        const float mTimeStepLimitBySpeedFactor = 0.4f;
        const float mTimeStepLimitByForceFactor = 0.25f;

        void computeSubTimeStepsByCFL(
            CudaMultiSphYan16ParticlesPtr &particles,
            const size_t phaseNum,
            const float sphMass[MULTISPH_MAX_PHASE_NUM],
            const float dt,
            const float kernelRadius,
            float renderInterval);
    };

    typedef SharedPtr<CudaMultiWCSphYan16Solver> CudaMultiWCSphYan16SolverPtr;
} // namespace KIRI

#endif /* _CUDA_MULTIWCSPH_REN14_SOLVER_CUH_ */