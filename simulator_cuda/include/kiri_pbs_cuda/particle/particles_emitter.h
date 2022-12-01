/*
 * @Author: Xu.Wang 
 * @Date: 2020-07-23 23:28:25 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-07-24 20:21:12
 */

#ifndef _PARTICLES_EMITTER_H_
#define _PARTICLES_EMITTER_H_

#include <kiri_pbs_cuda/cuda_common.cuh>

class ParticlesEmitter
{
public:
    ParticlesEmitter() {}

    ~ParticlesEmitter() {}
};

typedef std::shared_ptr<ParticlesEmitter>
    ParticlesEmitterPtr;

#endif /* _PARTICLES_EMITTER_H_ */