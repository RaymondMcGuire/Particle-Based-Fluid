/*
 * @Author: Xu.WANG
 * @Date: 2020-09-29 18:05:50
 * @LastEditTime: 2021-02-08 20:13:10
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\cuda_sph_dem_struct.cuh
 */

#ifndef _CUDA_SPH_DEM_STRUCT_CUH_
#define _CUDA_SPH_DEM_STRUCT_CUH_

#pragma once

#include <kiri_pbs_cuda/cuda_common.cuh>

struct SphBoxVolumeParams
{
    float3 Lower;
    int3 BoxSize;
};

struct SphDynamicVolumeParams
{
    uint EmitNum;
    float EmitRadiusScale;
    float3 EmitVelocity;
    float3 EmitPosition;
    bool SPHVisable;
};

struct SphDynamicEmitterParams
{
    bool SquareShapedEmitter;
    bool CustomDefine = false;

    // circle shape
    float EmitRadius;

    // rectangle shape
    float EmitWidth;
    float EmitHeight;

    float3 EmitVelocity;
    float3 EmitPosition;

    bool SPHVisable;
};

struct SphDemDemoParams
{
    // searcher
    int3 GridSize;
    float CellSize;
    float3 LowestPoint;
    float3 HighestPoint;
    float3 WorldCenter;
    float3 WorldSize;
    uint NumOfGridCells;

    bool PolySearcher;

    // dem volume
    float DemRestDensity;

    float3 DemInitBoxLowestPoint;
    float3 DemInitBoxHighestPoint;
    int3 DemInitBoxSize;
    float3 DemInitBoxColor;
    float3 DemWetColor;

    bool ShapeSamplingEnable;
    bool ShapeSamplingOffsetForce = false;
    bool ShapeSamplingRampColorEnable = false;
    float3 ShapeSamplingOffset;

    // sph volume
    float SphRestMass;
    float SphRestDensity;
    float SphCoefViscosity;
    float SphCoefEta;
    float SphCoefKappa;

    bool SurfaceTensionAdhesionMode = false;

    float3 SphInitBoxLowestPoint;
    float3 SphInitBoxHighestPoint;
    int3 SphInitBoxSize;
    float3 SphInitBoxColor;

    // number of particles
    uint maxNumOfParticles;
    uint NumOfParticleGroups;
    uint numOfParticles;
    uint NumOfPhases;
    uint NumOfBoundaries;

    // radius
    float DemParticleRadius;
    float SphParticleRadius;

    float BoundaryParticleRadius;

    float AvgParticleRadius;
    float MinParticleRadius;
    float MaxParticleRadius;

    float SphKernelRadius;
    float Dt;

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
    float AdhesionCoeff;
    float AdhesionSatCoeff;

    // emitter
    float3 EmitterPosition;
    uint EmitterKernalParticlesNum;

    // app
    int SceneConfigDataIdx = 0;
    char ExportBgeoFolderName[32] = "default";
    bool ExportBgeoFile = false;
    bool EmitParticles = false;
    bool DemInitEnable = true;
    bool SPHVisuable = true;
    bool RunSim = false;
    bool RunFluidSim = false;
    bool RunOffline = false;

    // ssf
    bool SsfFluidTransparentMode;
    bool SsfSoildParticleMode;
};
#endif