/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-25 18:52:38 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-26 02:37:04
 */

#ifndef _KIRI_PBF_SYSTEM__H_
#define _KIRI_PBF_SYSTEM__H_

#include <kiri_pch.h>
#include <kiri_core/pbd/pbf_system_data.h>

class KiriPBFSystem
{
public:
    KiriPBFSystem();
    ~KiriPBFSystem();

    // -----------------Getter Method-----------------
    KiriPBFSystemDataPtr pbfSystemData() const;
    // -----------------Getter Method-----------------

    // -----------------Init Environment -----------------
    void addBoxFluidAndBoxBoundary(Array1<BoundingBox3F> fluid, BoundingBox3F boundary, bool bcc = false);
    // -----------------Init Environment -----------------

    // -----------------PBF Method -----------------
    void Update();
    // -----------------PBF Method -----------------

private:
    // -----------------Coefficient-----------------
    float _coefViscosity = 0.02f;
    Vector3F _gravity = Vector3F(0.0f, (float)kiri_math::kGravity, 0.0f);
    float _timeStep = 0.005f;
    size_t _maxIter = 5;
    // -----------------Coefficient-----------------

    PointGenerator3Ptr _pointsGen;
    KiriPBFSystemDataPtr _pbfSystemData;

    void calcExternalForces();
    void updateTimeStepSizeCFL(const float &minTimeStep, const float &maxTimeStep);

    // semi-implicit Euler time integration.
    void semiImplicitEuler();

    void constraintProjection();

    void velocityUpdateFirstOrder();

    void computeXSPHViscosity();

    void computeVorticityConfinement();

    // calculate particle's fluid density using SPH method.
    bool computeFluidDensity(
        const size_t &particleIndex,
        const size_t &numFluidParticle,
        const ConstArrayAccessor1<Vector3F> &position,
        const ConstArrayAccessor1<float> &mass,
        const std::vector<size_t> &neighors,
        const float &SphKernelRadius,
        const float &fluidDensity,
        float &density_err,
        float &density);

    // calculate particle's lagrange multiplier.
    bool computeLagrangeMultiplier(
        const size_t &particleIndex,
        const size_t &numFluidParticle,
        const ConstArrayAccessor1<Vector3F> &position,
        const ConstArrayAccessor1<float> &mass,
        const std::vector<size_t> &neighors,
        const float &density,
        const float &SphKernelRadius,
        const float &fluidDensity,
        float &lambda);

    // perform a density constraint.
    bool solveDensityConstraint(
        const size_t &particleIndex,
        const size_t &numFluidParticle,
        const ConstArrayAccessor1<Vector3F> &position,
        const ConstArrayAccessor1<float> &mass,
        const std::vector<size_t> &neighors,
        const ConstArrayAccessor1<float> &lamba,
        const float &SphKernelRadius,
        const float &fluidDensity,
        Vector3F &deltaPos);
};

typedef SharedPtr<KiriPBFSystem> KiriPBFSystemPtr;

#endif