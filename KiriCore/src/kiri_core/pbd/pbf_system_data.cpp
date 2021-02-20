/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-25 18:35:33 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-26 01:58:54
 */

#include <kiri_core/pbd/pbf_system_data.h>

KiriPBFSystemData::KiriPBFSystemData()
{
    _lambdaIdx = addScalarData();
    _densityIdx = addScalarData();

    _massIdx = addScalarData();
    _invMassIdx = addScalarData();

    _positionIdx = addVectorData();
    _velocityIdx = addVectorData();
    _accelerationIdx = addVectorData();

    _oldPositionIdx = addVectorData();
    _restPositionIdx = addVectorData();
    _lastPositionIdx = addVectorData();

    _deltaPositionIdx = addVectorData();

    _kernelRadius = 4.0f * mParticleRadius;
}

KiriPBFSystemData::~KiriPBFSystemData() {}

// --------------------------------Setter Method--------------------------------
void KiriPBFSystemData::resizeScalar(size_t idx, size_t num)
{
    _scalarDataList[idx].resize(num, 0.0f);
}

void KiriPBFSystemData::resizeVector(size_t idx, size_t num)
{
    _vectorDataList[idx].resize(num, Vector3F());
}

void KiriPBFSystemData::SetParticleRadius(float particleRadius)
{
    mParticleRadius = std::max(particleRadius, 0.0f);
}

void KiriPBFSystemData::SetKernelRadius(float SphKernelRadius)
{
    _kernelRadius = SphKernelRadius;
}

float KiriPBFSystemData::calcFluidMass() const
{
    float diameter = 2.0f * particleRadius();
    float volume = diameter * diameter * diameter * 0.8f;
    float mass = volume * _fluidDensity;
    return mass;
}

void KiriPBFSystemData::addParticles(const Array1Vec3F &fluidPosition, const Array1Vec3F &boundaryPosition)
{

    _numOfFluidParticles = fluidPosition.size();
    _numOfBoundaryParticles = boundaryPosition.size();
    mNumOfParticles = _numOfFluidParticles + _numOfBoundaryParticles;

    //KIRI_INFO << "particles num:" << NumOfParticles();
    //KIRI_INFO << "fluid particles num:" << numOfFluidParticles();
    //KIRI_INFO << "boundary particles num:" << numOfBoundaryParticles();

    //resize params
    resizeScalar(_massIdx, mNumOfParticles);
    resizeScalar(_invMassIdx, mNumOfParticles);
    resizeVector(_positionIdx, mNumOfParticles);
    resizeVector(_velocityIdx, mNumOfParticles);
    resizeVector(_accelerationIdx, mNumOfParticles);
    resizeVector(_oldPositionIdx, mNumOfParticles);
    resizeVector(_restPositionIdx, mNumOfParticles);
    resizeVector(_lastPositionIdx, mNumOfParticles);

    resizeScalar(_lambdaIdx, _numOfFluidParticles);
    resizeScalar(_densityIdx, _numOfFluidParticles);
    resizeVector(_deltaPositionIdx, _numOfFluidParticles);

    auto p = positions();
    auto v = velocities();
    auto a = accelerations();
    auto op = oldPositions();
    auto rp = restPositions();
    auto lp = lastPositions();

    auto m = masses();
    auto invm = invMasses();

    auto l = lambdas();
    auto d = densities();
    auto dp = deltaPositions();

    //calculate fluid mass
    float mass = calcFluidMass();
    float invMass = (mass != 0) ? (1.0f / mass) : 0.0f;

    // add fluid particles
    kiri_math::parallelFor(kiri_math::kZeroSize, _numOfFluidParticles,
                           [&](size_t i) {
                               p[i] = fluidPosition[i];
                               v[i] = Vector3F();
                               a[i] = Vector3F();
                               op[i] = fluidPosition[i];
                               lp[i] = fluidPosition[i];
                               rp[i] = fluidPosition[i];

                               m[i] = mass;
                               invm[i] = invMass;

                               l[i] = 0.0f;
                               d[i] = 0.0f;
                               dp[i] = Vector3F();
                           });

    // build boundary particles searcher
    buildNeighborSearcher(SphKernelRadius(), boundaryPosition);
    buildNeighborLists(SphKernelRadius(), boundaryPosition);

    const kiri_math::SphCubicKernel3F mKernel(SphKernelRadius());

    // add boundary particles
    kiri_math::parallelFor(
        kiri_math::kZeroSize, _numOfBoundaryParticles,
        [&](size_t i) {
            p[i + _numOfFluidParticles] = boundaryPosition[i];
            v[i + _numOfFluidParticles] = Vector3F();
            a[i + _numOfFluidParticles] = Vector3F();
            op[i + _numOfFluidParticles] = boundaryPosition[i];
            lp[i + _numOfFluidParticles] = boundaryPosition[i];
            rp[i + _numOfFluidParticles] = boundaryPosition[i];

            // calculate boundary mass
            const auto &neighbors = neighborLists()[i];
            float delta = mKernel.W_zero();
            for (size_t j : neighbors)
            {
                delta += mKernel(boundaryPosition[i] - boundaryPosition[j]);
            }
            delta = _fluidDensity / delta;
            m[i + _numOfFluidParticles] = delta;
            float invDelta = (delta != 0) ? (1.0f / delta) : 0.0f;
            invm[i + _numOfFluidParticles] = invDelta;
        });
}

// --------------------------------Setter Method--------------------------------

// --------------------------------Data Container--------------------------------

size_t KiriPBFSystemData::addScalarData(size_t size, float initialVal)
{
    size_t attrIdx = _scalarDataList.size();
    _scalarDataList.emplace_back(size, initialVal);
    return attrIdx;
}

size_t KiriPBFSystemData::addVectorData(size_t size, const Vector3F &initialVal)
{
    size_t attrIdx = _vectorDataList.size();
    _vectorDataList.emplace_back(size, initialVal);
    return attrIdx;
}

ConstArrayAccessor1<float> KiriPBFSystemData::scalarDataAt(
    size_t idx) const
{
    return _scalarDataList[idx].constAccessor();
}

ArrayAccessor1<float> KiriPBFSystemData::scalarDataAt(size_t idx)
{
    return _scalarDataList[idx].accessor();
}

ConstArrayAccessor1<Vector3F> KiriPBFSystemData::vectorDataAt(
    size_t idx) const
{
    return _vectorDataList[idx].constAccessor();
}

ArrayAccessor1<Vector3F> KiriPBFSystemData::vectorDataAt(size_t idx)
{
    return _vectorDataList[idx].accessor();
}
// --------------------------------Data Container--------------------------------

// --------------------------------Getter Method--------------------------------
ConstArrayAccessor1<float> KiriPBFSystemData::lambdas() const
{
    return scalarDataAt(_lambdaIdx);
}
ArrayAccessor1<float> KiriPBFSystemData::lambdas()
{
    return scalarDataAt(_lambdaIdx);
}

ConstArrayAccessor1<float> KiriPBFSystemData::densities() const
{
    return scalarDataAt(_densityIdx);
}
ArrayAccessor1<float> KiriPBFSystemData::densities()
{
    return scalarDataAt(_densityIdx);
}

ArrayAccessor1<float> KiriPBFSystemData::masses()
{
    return scalarDataAt(_massIdx);
}

ConstArrayAccessor1<float> KiriPBFSystemData::masses() const
{
    return scalarDataAt(_massIdx);
}

ArrayAccessor1<float> KiriPBFSystemData::invMasses()
{
    return scalarDataAt(_invMassIdx);
}

ConstArrayAccessor1<float> KiriPBFSystemData::invMasses() const
{
    return scalarDataAt(_invMassIdx);
}

ArrayAccessor1<Vector3F> KiriPBFSystemData::positions()
{
    return vectorDataAt(_positionIdx);
}

ConstArrayAccessor1<Vector3F> KiriPBFSystemData::positions() const
{
    return vectorDataAt(_positionIdx);
}

ConstArrayAccessor1<Vector3F> KiriPBFSystemData::velocities() const
{
    return vectorDataAt(_velocityIdx);
}

ArrayAccessor1<Vector3F> KiriPBFSystemData::velocities()
{
    return vectorDataAt(_velocityIdx);
}

ConstArrayAccessor1<Vector3F> KiriPBFSystemData::accelerations() const
{
    return vectorDataAt(_accelerationIdx);
}

ArrayAccessor1<Vector3F> KiriPBFSystemData::accelerations()
{
    return vectorDataAt(_accelerationIdx);
}

ArrayAccessor1<Vector3F> KiriPBFSystemData::oldPositions()
{
    return vectorDataAt(_oldPositionIdx);
}

ConstArrayAccessor1<Vector3F> KiriPBFSystemData::oldPositions() const
{
    return vectorDataAt(_oldPositionIdx);
}

ArrayAccessor1<Vector3F> KiriPBFSystemData::restPositions()
{
    return vectorDataAt(_restPositionIdx);
}

ConstArrayAccessor1<Vector3F> KiriPBFSystemData::restPositions() const
{
    return vectorDataAt(_restPositionIdx);
}

ArrayAccessor1<Vector3F> KiriPBFSystemData::lastPositions()
{
    return vectorDataAt(_lastPositionIdx);
}

ConstArrayAccessor1<Vector3F> KiriPBFSystemData::lastPositions() const
{
    return vectorDataAt(_lastPositionIdx);
}

ArrayAccessor1<Vector3F> KiriPBFSystemData::deltaPositions()
{
    return vectorDataAt(_deltaPositionIdx);
}

ConstArrayAccessor1<Vector3F> KiriPBFSystemData::deltaPositions() const
{
    return vectorDataAt(_deltaPositionIdx);
}

size_t KiriPBFSystemData::numOfFluidParticles() const
{
    return _numOfFluidParticles;
}

size_t KiriPBFSystemData::numOfBoundaryParticles() const
{
    return _numOfBoundaryParticles;
}

size_t KiriPBFSystemData::NumOfParticles() const
{
    return mNumOfParticles;
}

float KiriPBFSystemData::particleRadius() const
{
    return mParticleRadius;
}

float KiriPBFSystemData::SphKernelRadius() const
{
    return _kernelRadius;
}

float KiriPBFSystemData::fluidDensity() const
{
    return _fluidDensity;
}

ConstArrayAccessor1<Vector3F> KiriPBFSystemData::fluidPositions() const
{
    size_t n = numOfFluidParticles();
    Array1Vec3F fluid;
    fluid.resize(n);

    auto p = positions();
    kiri_math::parallelFor(kiri_math::kZeroSize, n,
                           [&](size_t i) {
                               fluid[i] = p[i];
                           });
    return fluid.constAccessor();
}
ConstArrayAccessor1<Vector3F> KiriPBFSystemData::boundaryPositions() const
{
    size_t nf = numOfFluidParticles();
    size_t nb = numOfBoundaryParticles();
    Array1Vec3F boundary;
    boundary.resize(nb);

    auto p = positions();
    kiri_math::parallelFor(kiri_math::kZeroSize, nb,
                           [&](size_t i) {
                               boundary[i] = p[nf + i];
                           });
    return boundary.constAccessor();
}

// --------------------------------Getter Method--------------------------------

// --------------------------------Neighbor Searcher Method--------------------------------
const Array1<Vector3D> KiriPBFSystemData::cvtArrayF2D(ConstArrayAccessor1<Vector3F> list)
{
    Array1<Vector3D> arrayD;
    for (size_t i = 0; i < list.size(); i++)
    {
        arrayD.append(Vector3D((double)list[i].x, (double)list[i].y, (double)list[i].z));
    }
    return arrayD;
}

const PointNeighborSearcher3Ptr &KiriPBFSystemData::neighborSearcher() const
{
    return _neighborSearcher;
}

void KiriPBFSystemData::SetNeighborSearcher(
    const PointNeighborSearcher3Ptr &newNeighborSearcher)
{
    _neighborSearcher = newNeighborSearcher;
}

const std::vector<std::vector<size_t>> &
KiriPBFSystemData::neighborLists() const
{
    return _neighborLists;
}

void KiriPBFSystemData::buildNeighborSearcher(double maxSearchRadius, ConstArrayAccessor1<Vector3F> list)
{

    // Use PointParallelHashGridSearcher3 by default
    _neighborSearcher = std::make_shared<PointParallelHashGridSearcher3>(
        kDefaultHashGridResolution,
        kDefaultHashGridResolution,
        kDefaultHashGridResolution,
        2.0 * maxSearchRadius);

    _neighborSearcher->build(cvtArrayF2D(list).constAccessor());
}

void KiriPBFSystemData::buildNeighborLists(double maxSearchRadius, ConstArrayAccessor1<Vector3F> list)
{
    _neighborLists.resize(list.size());

    auto points = list;
    for (size_t i = 0; i < list.size(); ++i)
    {
        Vector3D origin((double)points[i].x, (double)points[i].y, (double)points[i].z);
        _neighborLists[i].clear();

        _neighborSearcher->forEachNearbyPoint(
            origin,
            maxSearchRadius,
            [&](size_t j, const Vector3D &) {
                if (i != j)
                {
                    _neighborLists[i].push_back(j);
                }
            });
    }
}

// --------------------------------Neighbor Searcher Method--------------------------------