/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-25 19:08:15 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-26 02:37:07
 */
#include <kiri_core/pbd/pbf_system.h>

KiriPBFSystem::KiriPBFSystem()
{
    _pbfSystemData = std::make_shared<KiriPBFSystemData>();
}

KiriPBFSystem::~KiriPBFSystem() {}

KiriPBFSystemDataPtr KiriPBFSystem::pbfSystemData() const
{
    return _pbfSystemData;
}

void KiriPBFSystem::addBoxFluidAndBoxBoundary(Array1<BoundingBox3F> fluids, BoundingBox3F boundary, bool bcc)
{
    float spacing = pbfSystemData()->particleRadius() * 2.0f;
    Array1Vec3F fluidPositions;
    Array1Vec3F boundaryPositions;
    //add fluid
    _pointsGen = std::make_shared<BccLatticePointGenerator>();
    if (bcc)
    {
        for (size_t i = 0; i < fluids.size(); i++)
        {
            _pointsGen->forEachPoint(fluids[i], spacing, 0, [&](const Vector3F &point) {
                fluidPositions.append(point);
                return true;
            });
        }
    }
    else
    {
        for (size_t i = 0; i < fluids.size(); i++)
        {
            _pointsGen->forEachPointWithNoOffset(fluids[i], spacing, [&](const Vector3F &point) {
                fluidPositions.append(point);
                return true;
            });
        }
    }

    //add boundary
    _pointsGen = std::make_shared<kiri_math::BboxSurfacePointGenerator>();
    _pointsGen->forEachPoint(boundary, spacing, 0, [&](const Vector3F &point) {
        boundaryPositions.append(point);
        return true;
    });

    pbfSystemData()->addParticles(fluidPositions, boundaryPositions);
}

// --------------------------------PBF Method--------------------------------

void KiriPBFSystem::Update()
{
    calcExternalForces();
    updateTimeStepSizeCFL(0.0001f, 0.005f);
    semiImplicitEuler();

    constraintProjection();
    velocityUpdateFirstOrder();

    //compute viscoity
    computeXSPHViscosity();
    computeVorticityConfinement();
}

// --------------------------------PBF Method--------------------------------
// --------------------------------PBF Calculation--------------------------------

void KiriPBFSystem::calcExternalForces()
{
    size_t n = pbfSystemData()->numOfFluidParticles();
    auto acceleration = pbfSystemData()->accelerations();
    auto m = pbfSystemData()->masses();

    kiri_math::parallelFor(
        kiri_math::kZeroSize,
        n,
        [&](size_t i) {
            // Gravity
            if (m[i] != 0.0f)
            {
                acceleration[i] = _gravity;
            }
        });
}

void KiriPBFSystem::updateTimeStepSizeCFL(const float &minTimeStep, const float &maxTimeStep)
{
    size_t n = pbfSystemData()->numOfFluidParticles();
    const float cflFactor = 1.0f;
    float timeStep = _timeStep;

    // Approximate max. position change due to current velocities
    float maxVelocity = 0.1f;
    auto v = pbfSystemData()->velocities();
    auto acc = pbfSystemData()->accelerations();
    const float diameter = 2.0f * pbfSystemData()->particleRadius();
    for (size_t i = 0; i < n; i++)
    {
        const float velMag = pow((v[i] + acc[i] * timeStep).length(), 2.0f);
        if (velMag > maxVelocity)
            maxVelocity = velMag;
    }

    timeStep = cflFactor * 0.40f * (diameter / sqrt(maxVelocity));
    timeStep = std::min(timeStep, maxTimeStep);
    timeStep = std::max(timeStep, minTimeStep);

    _timeStep = timeStep;
}

void KiriPBFSystem::semiImplicitEuler()
{
    size_t n = pbfSystemData()->numOfFluidParticles();
    auto op = pbfSystemData()->oldPositions();
    auto lp = pbfSystemData()->lastPositions();

    auto p = pbfSystemData()->positions();
    auto v = pbfSystemData()->velocities();
    auto acc = pbfSystemData()->accelerations();

    auto dp = pbfSystemData()->deltaPositions();

    auto m = pbfSystemData()->masses();

    kiri_math::parallelFor(
        kiri_math::kZeroSize,
        n,
        [&](size_t i) {
            dp[i] = Vector3F();
            lp[i] = op[i];
            op[i] = p[i];

            if (m[i] != 0)
            {
                v[i] += acc[i] * _timeStep;
                p[i] += v[i] * _timeStep;
            }
        });
}

void KiriPBFSystem::constraintProjection()
{
    size_t n = pbfSystemData()->numOfFluidParticles();

    auto p = pbfSystemData()->positions();
    auto m = pbfSystemData()->masses();
    auto d = pbfSystemData()->densities();
    auto l = pbfSystemData()->lambdas();

    auto dp = pbfSystemData()->deltaPositions();

    auto kr = pbfSystemData()->SphKernelRadius();
    auto fd = pbfSystemData()->fluidDensity();
    auto fp = pbfSystemData()->fluidPositions();

    // build fluid particles searcher
    pbfSystemData()->buildNeighborSearcher(kr, p);
    pbfSystemData()->buildNeighborLists(kr, p);
    size_t iter = 0;
    while (iter < _maxIter)
    {
        // calculate density and lagrange multiplier.
        kiri_math::parallelFor(
            kiri_math::kZeroSize,
            n,
            [&](size_t i) {
                float density_err;
                const auto &neighbors = pbfSystemData()->neighborLists()[i];
                computeFluidDensity(i, n, p, m, neighbors, kr, fd, density_err, d[i]);
                computeLagrangeMultiplier(i, n, p, m, neighbors, d[i], kr, fd, l[i]);
            });

        // perform density constraint.
        kiri_math::parallelFor(
            kiri_math::kZeroSize,
            n,
            [&](size_t i) {
                const auto &neighbors = pbfSystemData()->neighborLists()[i];
                solveDensityConstraint(i, n, p, m, neighbors, l, kr, fd, dp[i]);
            });

        // add the delta position to particles' position.
        kiri_math::parallelFor(
            kiri_math::kZeroSize,
            n,
            [&](size_t i) {
                p[i] += dp[i];
            });

        ++iter;
    }
}

void KiriPBFSystem::velocityUpdateFirstOrder()
{
    size_t n = pbfSystemData()->numOfFluidParticles();
    auto p = pbfSystemData()->positions();
    auto m = pbfSystemData()->masses();
    auto v = pbfSystemData()->velocities();
    auto op = pbfSystemData()->oldPositions();
    // update velocities.
    kiri_math::parallelFor(
        kiri_math::kZeroSize,
        n,
        [&](size_t i) {
            if (m[i] != 0.0f)
            {
                v[i] = (1.0f / _timeStep) * (p[i] - op[i]);
            }
        });
}

void KiriPBFSystem::computeXSPHViscosity()
{
    size_t n = pbfSystemData()->numOfFluidParticles();

    auto p = pbfSystemData()->positions();
    auto m = pbfSystemData()->masses();
    auto v = pbfSystemData()->velocities();
    auto d = pbfSystemData()->densities();
    float SphKernelRadius = pbfSystemData()->SphKernelRadius();
    const kiri_math::SphCubicKernel3F mKernel(SphKernelRadius);

    kiri_math::parallelFor(
        kiri_math::kZeroSize,
        n,
        [&](size_t i) {
            const auto &neighbors = pbfSystemData()->neighborLists()[i];
            Vector3F sum_value(0.0f);
            for (size_t j : neighbors)
            {
                if (j < n)
                {

                    Vector3F tmp = v[i] - v[j];
                    tmp *= mKernel(p[i] - p[j]) * (m[j] / d[j]);
                    sum_value -= tmp;
                }
            }
            sum_value *= _coefViscosity;
            v[i] += sum_value;
        });
}

void KiriPBFSystem::computeVorticityConfinement()
{
    size_t n = pbfSystemData()->numOfFluidParticles();

    auto p = pbfSystemData()->positions();
    auto m = pbfSystemData()->masses();
    auto v = pbfSystemData()->velocities();
    auto d = pbfSystemData()->densities();

    float SphKernelRadius = pbfSystemData()->SphKernelRadius();
    const kiri_math::SphCubicKernel3F mKernel(SphKernelRadius);

    Array1Vec3F deltaVelocity;
    deltaVelocity.resize(n);

    kiri_math::parallelFor(
        kiri_math::kZeroSize,
        n,
        [&](size_t i) {
            const auto &neighbors = pbfSystemData()->neighborLists()[i];

            Vector3F N(0.0f);
            Vector3F curl(0.0f);
            Vector3F curlX(0.0f);
            Vector3F curlY(0.0f);
            Vector3F curlZ(0.0f);

            for (size_t j : neighbors)
            {
                if (j >= n)
                    continue;
                const Vector3F velGap = v[j] - v[i];

                curl += velGap.cross(mKernel.gradW(p[i] - p[j]));
                curlX += velGap.cross(mKernel.gradW(p[i] + Vector3F(0.01f, 0.0f, 0.0f) - p[j]));
                curlY += velGap.cross(mKernel.gradW(p[i] + Vector3F(0.0f, 0.01f, 0.0f) - p[j]));
                curlZ += velGap.cross(mKernel.gradW(p[i] + Vector3F(0.0f, 0.0f, 0.01f) - p[j]));
            }

            if (curl.x == curl.x || curl.y == curl.y || curl.z == curl.z)
            {
                float curlLen = curl.length();
                N.x = curlX.length() - curlLen;
                N.y = curlY.length() - curlLen;
                N.z = curlZ.length() - curlLen;
                N.normalize();

                if (N.x == N.x || N.y == N.y || N.z == N.z)
                {
                    Vector3F force = 0.000010f * N.cross(curl);
                    deltaVelocity[i] = _timeStep * force;
                }
            }
        });

    kiri_math::parallelFor(
        kiri_math::kZeroSize,
        n,
        [&](size_t i) {
            v[i] += deltaVelocity[i];
        });
}

bool KiriPBFSystem::computeFluidDensity(
    const size_t &particleIndex,
    const size_t &numFluidParticle,
    const ConstArrayAccessor1<Vector3F> &position,
    const ConstArrayAccessor1<float> &mass,
    const std::vector<size_t> &neighbors,
    const float &SphKernelRadius,
    const float &fluidDensity,
    float &density_err,
    float &density)
{
    const kiri_math::SphCubicKernel3F mKernel(SphKernelRadius);

    density = mass[particleIndex] * mKernel.W_zero();
    for (size_t j : neighbors)
    {
        if (j < numFluidParticle)
        {
            density += mass[j] * mKernel(position[particleIndex] - position[j]);
        }
        else
        {
            density += mass[j] * mKernel(position[particleIndex] - position[j]);
        }
    }

    density_err = std::max(density, fluidDensity) - fluidDensity;
    return true;
}

bool KiriPBFSystem::computeLagrangeMultiplier(
    const size_t &particleIndex,
    const size_t &numFluidParticle,
    const ConstArrayAccessor1<Vector3F> &position,
    const ConstArrayAccessor1<float> &mass,
    const std::vector<size_t> &neighbors,
    const float &density,
    const float &SphKernelRadius,
    const float &fluidDensity,
    float &lambda)
{
    const kiri_math::SphCubicKernel3F mKernel(SphKernelRadius);

    const float eps = 1.0e-6f;
    const float constraint = std::max(density / fluidDensity - 1.0f, 0.0f);
    if (constraint != 0.0f)
    {
        float sum_grad_cj = 0.0f;
        Vector3F grad_ci(0.0f);
        for (size_t j : neighbors)
        {
            if (j < numFluidParticle)
            {
                Vector3F grad_cj = mass[j] / fluidDensity * mKernel.gradW(position[particleIndex] - position[j]);
                sum_grad_cj += pow(grad_cj.length(), 2.0f);
                grad_ci += grad_cj;
            }
            else
            {
                Vector3F grad_cj = mass[j] / fluidDensity * mKernel.gradW(position[particleIndex] - position[j]);
                sum_grad_cj += pow(grad_cj.length(), 2.0f);
                grad_ci += grad_cj;
            }
        }
        sum_grad_cj += pow(grad_ci.length(), 2.0f);
        lambda = -constraint / (sum_grad_cj + eps);
    }
    else
        lambda = 0.0f;

    return true;
}

bool KiriPBFSystem::solveDensityConstraint(
    const size_t &particleIndex,
    const size_t &numFluidParticle,
    const ConstArrayAccessor1<Vector3F> &position,
    const ConstArrayAccessor1<float> &mass,
    const std::vector<size_t> &neighbors,
    const ConstArrayAccessor1<float> &lambda,
    const float &SphKernelRadius,
    const float &fluidDensity,
    Vector3F &deltaPos)
{
    const kiri_math::SphCubicKernel3F mKernel(SphKernelRadius);

    deltaPos = Vector3F(0.0f);

    for (size_t j : neighbors)
    {
        if (j < numFluidParticle)
        {
            Vector3F grad_cj = mass[j] / fluidDensity * mKernel.gradW(position[particleIndex] - position[j]);
            deltaPos += (lambda[particleIndex] + lambda[j]) * grad_cj;
        }
        else
        {
            Vector3F grad_cj = mass[j] / fluidDensity * mKernel.gradW(position[particleIndex] - position[j]);
            deltaPos += (lambda[particleIndex]) * grad_cj;
        }
    }

    return true;
}

// --------------------------------PBF Calculation--------------------------------
