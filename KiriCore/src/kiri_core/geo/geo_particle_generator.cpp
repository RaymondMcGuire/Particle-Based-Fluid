/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-21 23:19:12
 * @LastEditTime: 2020-11-24 15:34:42
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\geo\geo_particle_generator.cpp
 */
#include <kiri_core/geo/geo_particle_generator.h>
#include <random>
#include <omp.h>
void KiriGeoParticleGenerator::generateParticles()
{
    float spacing = mParticleRadius * 2.f * mSamplingRatio;
    float jitter = mJitterRatio * mParticleRadius;

    auto boxMin = obj->getAABBMin();
    auto boxMax = obj->getAABBMax();
    KIRI_LOG_DEBUG("mParticleRadius={0}", mParticleRadius);
    KIRI_LOG_DEBUG("boxMin=({0},{1},{2})", boxMin.x, boxMin.y, boxMin.z);
    KIRI_LOG_DEBUG("boxMax=({0},{1},{2})", boxMax.x, boxMax.y, boxMax.z);

    Vector3F grid = (boxMax - boxMin) / spacing;

    for (Int d = 0; d < grid.size(); ++d)
    {
        grid[d] = ceil(grid[d]);
    }

    //KIRI_LOG_INFO("Grid Size=({0:f},{1:f},{2:f})", grid.x, grid.y, grid.z);

    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::uniform_real_distribution<> dist(-1.f, 1.f);

    omp_lock_t writelock;
    omp_init_lock(&writelock);

    UInt count = 0;
    float maxNum = grid[0] * grid[1] * grid[2];
    kiri_math::parallelFor<UInt>(0, static_cast<UInt>(grid[0]), 0, static_cast<UInt>(grid[1]), 0, static_cast<UInt>(grid[2]),
                                 [&](UInt i, UInt j, UInt k) {
                                     Vector3F ppos = boxMin + Vector3F((float)i, (float)j, (float)k) * spacing;
                                     auto geoPhi = obj->signedDistance(ppos);

                                     if (geoPhi < 0)
                                     {
                                         if (geoPhi < -jitter)
                                         {
                                             ppos += jitter * Vector3F(dist(engine), dist(engine), dist(engine)).normalized();
                                         }

                                         omp_set_lock(&writelock);

                                         particles.append(Vector4F(ppos.x, ppos.y, ppos.z, mParticleRadius));

                                         omp_unset_lock(&writelock);
                                     }
                                 });

    KIRI_LOG_INFO("Sampling Number={0:d}", particles.size());
}