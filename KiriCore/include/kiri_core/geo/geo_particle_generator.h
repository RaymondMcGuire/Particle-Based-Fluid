/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-21 23:07:07
 * @LastEditTime: 2021-02-20 19:45:28
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\geo\geo_particle_generator.h
 */

#ifndef _KIRI_GEO_PARTICLE_GENERATOR_H_
#define _KIRI_GEO_PARTICLE_GENERATOR_H_
#pragma once
#include <kiri_core/geo/geo_object.h>

class KiriGeoParticleGenerator
{
public:
    KiriGeoParticleGenerator() = default;
    KiriGeoParticleGenerator(String Name, float ParticleRadius, float SamplingRatio, float JitterRatio, Vector3F Offset = Vector3F(0.f), float BoxScale = 1.f)
        : mParticleRadius(ParticleRadius), mSamplingRatio(SamplingRatio), mJitterRatio(JitterRatio)
    {
        obj = std::make_shared<KiriTriMeshObject>(Name, ParticleRadius, Offset, BoxScale);
        generateParticles();
    };

    Array1Vec4F particles;

private:
    void generateParticles();

    float mParticleRadius;
    float mSamplingRatio;
    float mJitterRatio;

    KiriTriMeshObjectPtr obj;
};

typedef SharedPtr<KiriGeoParticleGenerator> KiriGeoParticleGeneratorPtr;
#endif
