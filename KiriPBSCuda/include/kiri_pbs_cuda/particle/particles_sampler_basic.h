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

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>
class ParticlesSamplerBasic
{
public:
    ParticlesSamplerBasic();
    std::vector<float3> GetBoxSampling(float3 lower, float3 upper, float spacing);

private:
    std::vector<float3> mPoints;
};

typedef std::shared_ptr<ParticlesSamplerBasic> ParticlesSamplerBasicPtr;

#endif