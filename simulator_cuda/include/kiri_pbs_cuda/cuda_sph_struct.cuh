/*
 * @Author: Xu.Wang
 * @Date: 2020-06-13 00:10:51
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-07-24 20:50:45
 */

#ifndef _CUDA_SPH_STRUCT_CUH_
#define _CUDA_SPH_STRUCT_CUH_

#include <kiri_pbs_cuda/cuda_common.cuh>

struct BoxParticlesParams
{
    float3 startVelocity;

    float3 boxes_lower;
    float3 boxes_upper;
    int3 boxes_size;
    float3 boxes_color;

    int phase_type;
    float rest_density;
    float rest_mass;
    float coef_viscosity;
};

struct SquareVolumeParticlesParams
{
    float3 startVelocity;

    float3 lower_point;
    float3 upper_point;
    int3 size;
    float3 color;

    int phase_number;
    int phase_type;
    float rest_density;
    float rest_mass;
    float coef_viscosity;
};

struct ThreePhaseParamsRen2014
{
    float mixtureDensity;

    float mixturePressure;
    float kPhasePressures[3];
    float3 gradientPressures[3];

    float volumeFractions[3];
    float oldVolumeFractions[3];
    float3 gradientVolumeFractions[3];
    float deltaVolumeFractions[3];

    float3 driftVelocities[3];
};

struct Ren2014Params
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
    bool miscible;
    float tou;
    float eta;

    float maxVel;

    // particle
    int NumOfPhases;
    uint NumOfBoundaries;
    uint numOfParticles;
    uint numOfTotalParticles;

    // sph params
    float particleRadius;
    float sphRadius;
    float dt;
    float kappa;

    thrust::host_vector<float> rest_density;
    thrust::host_vector<float> rest_mass;
    thrust::host_vector<float> coef_viscosity;
};

struct TwoPhaseParamsYan2016
{
    uint phaseType;
    uint ophaseType;
    uint phaseTypes[3];
    float mixtureDensity;

    float mixturePressure;
    float kPhasePressures[3];
    float3 gradientPressures[3];

    float volumeFractions[3];
    float oldVolumeFractions[3];
    float3 gradientVolumeFractions[3];
    float deltaVolumeFractions[3];

    float3 driftVelocities[3];

    // solid
    tensor3x3 stressTensor[3];
    tensor3x3 stressRateTensor[3];
    tensor3x3 deviatoricStressTensor[3];
    tensor3x3 deviatoricStressRateTensor[3];
};

struct Yan2016Params
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
    bool miscible;
    float tou;
    float eta;

    float maxVel;

    // particle
    int NumOfPhases;
    uint NumOfBoundaries;
    uint numOfParticles;
    uint numOfTotalParticles;
    float3 startVelocity;

    // sph params
    float particleRadius;
    float sphRadius;
    float dt;
    float kappa;

    // 0: fluid 1: elastoplastic 2: hypoplastic
    thrust::host_vector<uint> phase_type;
    thrust::host_vector<float> rest_density;
    thrust::host_vector<float> rest_mass;
    thrust::host_vector<float> coef_viscosity;
};
#endif