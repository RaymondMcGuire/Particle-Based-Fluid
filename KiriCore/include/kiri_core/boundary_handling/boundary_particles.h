/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 00:18:57
 * @LastEditTime: 2021-02-20 19:44:40
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\boundary_handling\boundary_particles.h
 */

#ifndef _KIRI_BOUNDARY_PARTICLES_H_
#define _KIRI_BOUNDARY_PARTICLES_H_
#pragma once
#include <kiri_pch.h>

class KiriBoundaryParticles
{
public:
    KiriBoundaryParticles(BoundingBox3F bbox, float particleRadius);

    size_t numOfBoundaryParticles() { return _numOfBoundaryParticles; }
    Array1Vec3F boundaryParticles() { return _boundaryParticles; }

protected:
    Array1Vec3F _boundaryParticles;
    size_t _numOfBoundaryParticles;

    BoundingBox3F _bbox;
    float mParticleRadius;

    //BboxSurfacePointGeneratorPtr _pointsGen;
};
typedef SharedPtr<KiriBoundaryParticles> KiriBoundaryParticlesPtr;
#endif