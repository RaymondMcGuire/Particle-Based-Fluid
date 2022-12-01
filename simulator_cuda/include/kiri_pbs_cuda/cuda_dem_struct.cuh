/*
 * @Author: Xu.WANG
 * @Date: 2020-07-21 16:37:22
 * @LastEditTime: 2021-02-08 20:13:24
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\cuda_dem_struct.cuh
 */

#ifndef _CUDA_DEM_STRUCT_CUH_
#define _CUDA_DEM_STRUCT_CUH_

#pragma once

#include <kiri_pbs_cuda/cuda_common.cuh>

#define MSM_MAX_SUB_SPHERE_NUM 8

enum MSM_TYPE
{
    MSM_S1 = 1,
    MSM_L2 = 2,
    MSM_L3 = 3,
    MSM_C8 = 4,
    MSM_M7 = 5,
    MSM_T4 = 6
};

struct DEMSphere
{
    int clumpId;
    float radius;
    float3 center;
    float3 color;
    DEMSphere(const float3 &_c, float _r, float3 _color = make_float3(0.f), int _clumpId = -1)
        : center(_c), radius(_r), color(_color), clumpId(_clumpId){};
};

struct DEMClumpInfo
{
    uint clumpId;
    int subId;
    float3 relPos;
    quaternion relOri;
};

struct DEMClump
{
    int clumpId;

    float3 centroid;
    quaternion ori;
    float3 vel;
    float3 angVel;
    float3 angMom;
    float3 inertia;
    float mass;

    int subNum;
    float3 force[MSM_MAX_SUB_SPHERE_NUM];
    float3 torque[MSM_MAX_SUB_SPHERE_NUM];

    float3 subColor[MSM_MAX_SUB_SPHERE_NUM];
    float subRadius[MSM_MAX_SUB_SPHERE_NUM];
    float3 subPos[MSM_MAX_SUB_SPHERE_NUM];
    quaternion subOri[MSM_MAX_SUB_SPHERE_NUM];
};

struct DEMClumpForGenerator
{
    int clumpId;
    float radius;
    float3 center;
    int minId, maxId;
};

struct DEMVolumeParticlesParams
{
    float particleRadius;

    float3 lower_point;
    float3 upper_point;
    int3 size;
    float3 color;
};

struct DEMDemoParams
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

    // number of particles
    uint NumOfParticleGroups;
    uint numOfParticles;
    uint NumOfPhases;
    uint NumOfBoundaries;

    // radius
    float AvgParticleRadius;
    float MinParticleRadius;
    float MaxParticleRadius;

    // sph params
    float sphRadius;
    float Dt;

    thrust::host_vector<float> rest_density;
    // dem params
    float sr;

    bool Spherical;
    // dem params
    float DemCoefDamping;
    float DemCoefYoung;
    float DemCoefPoisson;
    float TangensOfFrictionAngle;

    // seepage params
    float C0;
    float Csat;
    float DragCoeff;
};

#endif