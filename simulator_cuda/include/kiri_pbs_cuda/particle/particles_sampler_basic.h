/*** 
 * @Author: Pierre-Luc Manteaux
 * @Date: 2020-06-16 01:32:28
 * @LastEditTime: 2021-02-08 20:00:52
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\particles_sampler_basic.h
 * @Reference: https://github.com/manteapi/hokusai
 */

#ifndef _PARTICLES_SAMPLER_BASIC_H_
#define _PARTICLES_SAMPLER_BASIC_H_

#pragma once

#include <kiri_pbs_cuda/cuda_common.cuh>
#include <kiri_pbs_cuda/solver/dem/msm_pack.h>
class ParticlesSamplerBasic
{
public:
    ParticlesSamplerBasic();
    std::vector<float3> GetBoxSampling(float3 lower, float3 upper, float spacing);

    std::vector<DEMSphere> GetCloudSampling(float3 lower, float3 upper, float rMean, float rFuzz = 0.f, int maxNumOfParticles = -1);

    std::vector<DEMClump> GetRndClumpCloudSampling(float3 lower, float3 upper, const std::vector<MSMPackPtr> &clumpTypes, float rMean, float rFuzz = 0.f, int maxNumOfClumps = -1);
    std::vector<DEMClump> GetCDFClumpCloudSampling(float3 lower, float3 upper, const std::vector<MSMPackPtr> clumpTypes, const std::vector<float> clumpTypesProb, const std::vector<float> radiusRange, const std::vector<float> radiusConstantProb, int maxNumOfClumps = -1);

    std::vector<DEMSphere> GetPack() { return mPack; }

private:
    std::vector<float3> mPoints;
    std::vector<DEMSphere> mPack;
};

typedef std::shared_ptr<ParticlesSamplerBasic> ParticlesSamplerBasicPtr;

#endif