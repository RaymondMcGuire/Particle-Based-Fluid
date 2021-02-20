/*
 * @Author: Xu.Wang 
 * @Date: 2020-05-29 11:23:23 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-29 23:11:48
 */

#include <kiri_core/boundary_handling/boundary_particles.h>

KiriBoundaryParticles::KiriBoundaryParticles(BoundingBox3F bbox, float particleRadius)
{
    //_bbox = bbox;
    //mParticleRadius = particleRadius;

    //_pointsGen = std::make_shared<BboxSurfacePointGenerator>();
    //_pointsGen->forEachPointNoTop(_bbox, 2.f * mParticleRadius, 3.f, [&](const Vector3F &point) {
    //    _boundaryParticles.append(point);
    //    return true;
    //});

    //_numOfBoundaryParticles = _boundaryParticles.size();

    //KIRI_INFO << "Boundary Particles Num:" << numOfBoundaryParticles() << ", Boundary Particles Radius:" << mParticleRadius;
}