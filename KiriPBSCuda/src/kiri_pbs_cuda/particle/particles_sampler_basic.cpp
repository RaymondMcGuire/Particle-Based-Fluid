#include <random>
#include <list>
#include <kiri_pbs_cuda/particle/particles_sampler_basic.h>
#include <Eigen/Eigenvalues>

ParticlesSamplerBasic::ParticlesSamplerBasic()
{
}

std::vector<float3> ParticlesSamplerBasic::GetBoxSampling(float3 lower, float3 upper, float spacing)
{
    mPoints.clear();

    int epsilon = 0;
    float3 sides = (upper - lower) / spacing;

    //ZX plane - bottom
    for (int i = -epsilon; i <= sides.x + epsilon; ++i)
    {
        for (int j = -epsilon; j <= sides.z + epsilon; ++j)
        {
            mPoints.push_back(make_float3(lower.x + i * spacing, lower.y, lower.z + j * spacing));
        }
    }

    //ZX plane - top
    for (int i = -epsilon; i <= sides.x + epsilon; ++i)
    {
        for (int j = -epsilon; j <= sides.z + epsilon; ++j)
        {
            mPoints.push_back(make_float3(lower.x + i * spacing, upper.y, lower.z + j * spacing));
        }
    }

    //XY plane - back
    for (int i = -epsilon; i <= sides.x + epsilon; ++i)
    {
        for (int j = -epsilon; j <= sides.y + epsilon; ++j)
        {
            mPoints.push_back(make_float3(lower.x + i * spacing, lower.y + j * spacing, lower.z));
        }
    }

    //XY plane - front
    for (int i = -epsilon; i <= sides.x + epsilon; ++i)
    {
        for (int j = -epsilon; j <= sides.y - epsilon; ++j)
        {
            mPoints.push_back(make_float3(lower.x + i * spacing, lower.y + j * spacing, upper.z));
        }
    }

    //YZ plane - left
    for (int i = -epsilon; i <= sides.y + epsilon; ++i)
    {
        for (int j = -epsilon; j <= sides.z + epsilon; ++j)
        {
            mPoints.push_back(make_float3(lower.x, lower.y + i * spacing, lower.z + j * spacing));
        }
    }

    //YZ plane - right
    for (int i = -epsilon; i <= sides.y + epsilon; ++i)
    {
        for (int j = -epsilon; j <= sides.z + epsilon; ++j)
        {
            mPoints.push_back(make_float3(upper.x, lower.y + i * spacing, lower.z + j * spacing));
        }
    }
    return mPoints;
}