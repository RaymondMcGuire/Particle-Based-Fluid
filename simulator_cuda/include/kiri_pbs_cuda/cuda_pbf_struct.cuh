/*
 * @Author: Xu.Wang
 * @Date: 2020-06-07 12:41:27
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-06-13 11:17:32
 */

#ifndef _CUDA_PBF_STRUCT_CUH_
#define _CUDA_PBF_STRUCT_CUH_

#include <kiri_pbs_cuda/cuda_common.cuh>

struct UpdateVelocity
{
    float inv_dt;
    UpdateVelocity(float dt) : inv_dt(1.f / dt) {}

    template <typename T>
    __device__ float3 operator()(T t)
    {
        float3 pos = thrust::get<0>(t), npos = thrust::get<1>(t);
        return (npos - pos) * inv_dt;
    }
};

struct PBFParams
{
    // searcher
    int3 GridSize;
    float CellSize;
    float3 LowestPoint;
    float3 HighestPoint;
    uint NumOfGridCells;

    // sph
    uint numOfParticles;
    float particleRadius;
    float sphRadius;
    float restDensity;

    // pbf
    float dt;
    int maxIterNums;
    float deltaQ;
    float lambdaEps;
    float coefXSPH;
    float sCorrN;
    float sCorrK;

    // init particles
    int boxParticleType;
};

struct TwoPhaseParamsYang2015
{
    float aggregateDensity;

    float massRatios[2];
    float oldMassRatios[2];
    float deltaMassRatios[2];
    float massRatioLaplacians[2];
    float massRatioNormLaplacians[2];
    float3 massRatioGradient[2];
    float chemicalPotentials[2];
    float3 chemicalPotentialGradient[2];
};

struct Yang2015Params
{
    // searcher
    int3 GridSize;
    float CellSize;
    float3 LowestPoint;
    float3 HighestPoint;
    uint NumOfGridCells;

    // scene configuration
    thrust::host_vector<float3> boxes_lower;
    thrust::host_vector<float3> boxes_upper;
    thrust::host_vector<int3> boxes_size;
    thrust::host_vector<float3> boxes_color;

    // multi
    float mobilities;
    float eta;
    float alpha;
    float s1;
    float s2;
    float epsilon;
    float sigma;

    // particle
    int NumOfPhases;
    uint NumOfBoundaries;
    uint numOfParticles;
    uint numOfTotalParticles;
    thrust::host_vector<float> rest_density;
    thrust::host_vector<float> particle_mass;

    float particleRadius;
    float sphRadius;

    // pbf params
    float dt;
    int maxIterNums;
    float deltaQ;
    float lambdaEps;
    float coefXSPH;
    float sCorrN;
    float sCorrK;
};

#endif