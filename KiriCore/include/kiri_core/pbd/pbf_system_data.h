/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-25 18:22:57 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-26 02:40:38
 */

#ifndef _KIRI_PBF_SYSTEM_DATA_H_
#define _KIRI_PBF_SYSTEM_DATA_H_

#include <kiri_pch.h>

class KiriPBFSystemData
{
public:
    KiriPBFSystemData();
    ~KiriPBFSystemData();

    // -----------------Data Container-----------------
    typedef Array1<float> ScalarData;
    typedef Array1Vec3F VectorData;

    size_t addScalarData(size_t size = 0, float initialVal = 0.0f);
    size_t addVectorData(size_t size = 0, const Vector3F &initialVal = Vector3F());

    ConstArrayAccessor1<float> scalarDataAt(size_t idx) const;
    ArrayAccessor1<float> scalarDataAt(size_t idx);
    ConstArrayAccessor1<Vector3F> vectorDataAt(size_t idx) const;
    ArrayAccessor1<Vector3F> vectorDataAt(size_t idx);
    // -----------------Data Container-----------------

    // -----------------Getter Method-----------------
    size_t numOfFluidParticles() const;
    size_t numOfBoundaryParticles() const;
    size_t NumOfParticles() const;

    float fluidDensity() const;

    ConstArrayAccessor1<Vector3F> fluidPositions() const;
    ConstArrayAccessor1<Vector3F> boundaryPositions() const;

    // pbf system data container
    ConstArrayAccessor1<float> lambdas() const;
    ArrayAccessor1<float> lambdas();

    ConstArrayAccessor1<float> densities() const;
    ArrayAccessor1<float> densities();

    ConstArrayAccessor1<float> masses() const;
    ArrayAccessor1<float> masses();
    ConstArrayAccessor1<float> invMasses() const;
    ArrayAccessor1<float> invMasses();

    ConstArrayAccessor1<Vector3F> positions() const;
    ArrayAccessor1<Vector3F> positions();
    ConstArrayAccessor1<Vector3F> velocities() const;
    ArrayAccessor1<Vector3F> velocities();
    ConstArrayAccessor1<Vector3F> accelerations() const;
    ArrayAccessor1<Vector3F> accelerations();

    ConstArrayAccessor1<Vector3F> oldPositions() const;
    ArrayAccessor1<Vector3F> oldPositions();
    ConstArrayAccessor1<Vector3F> restPositions() const;
    ArrayAccessor1<Vector3F> restPositions();
    ConstArrayAccessor1<Vector3F> lastPositions() const;
    ArrayAccessor1<Vector3F> lastPositions();

    ConstArrayAccessor1<Vector3F> deltaPositions() const;
    ArrayAccessor1<Vector3F> deltaPositions();

    float particleRadius() const;
    float SphKernelRadius() const;
    // -----------------Getter Method-----------------

    // -----------------Setter Method-----------------
    void SetParticleRadius(float particleRadius);

    void SetKernelRadius(float SphKernelRadius);

    void addParticles(const Array1Vec3F &fluidPosition, const Array1Vec3F &boundaryPosition);
    // -----------------Setter Method-----------------

    // -----------------Neighbor Searcher Method-----------------
    const PointNeighborSearcher3Ptr &neighborSearcher() const;

    void SetNeighborSearcher(
        const PointNeighborSearcher3Ptr &newNeighborSearcher);

    const std::vector<std::vector<size_t>> &neighborLists() const;
    void buildNeighborSearcher(double maxSearchRadius, ConstArrayAccessor1<Vector3F> list);
    void buildNeighborLists(double maxSearchRadius, ConstArrayAccessor1<Vector3F> list);
    // -----------------Neighbor Searcher Method-----------------

    // -----------------Data init-----------------
    float calcFluidMass() const;
    float calcBoundaryMass() const;
    // -----------------Data init-----------------
private:
    // -----------------Coefficient-----------------
    float _fluidDensity = kiri_math::kWaterDensity;
    // -----------------Coefficient-----------------

    // -----------------Data Container-----------------
    std::vector<ScalarData> _scalarDataList;
    std::vector<VectorData> _vectorDataList;

    size_t _lambdaIdx;
    size_t _densityIdx;
    size_t _deltaPositionIdx;

    size_t _massIdx;
    size_t _invMassIdx;

    size_t _positionIdx;
    size_t _velocityIdx;
    size_t _accelerationIdx;

    size_t _oldPositionIdx;
    size_t _restPositionIdx;
    size_t _lastPositionIdx;

    size_t _numOfFluidParticles = 0;
    size_t _numOfBoundaryParticles = 0;
    size_t mNumOfParticles = 0;
    float mParticleRadius = 0.017f;
    float _kernelRadius = 0.068f;
    // -----------------Data Container-----------------

    // -----------------Setter Method-----------------
    void resizeScalar(size_t idx, size_t num);
    void resizeVector(size_t idx, size_t num);

    // -----------------Setter Method-----------------

    // -----------------Neighbor Searcher Method-----------------
    size_t kDefaultHashGridResolution = 8;
    PointNeighborSearcher3Ptr _neighborSearcher;
    std::vector<std::vector<size_t>> _neighborLists;

    const Array1<Vector3D> cvtArrayF2D(ConstArrayAccessor1<Vector3F> list);
    // -----------------Neighbor Searcher Method-----------------
};

typedef SharedPtr<KiriPBFSystemData> KiriPBFSystemDataPtr;

#endif