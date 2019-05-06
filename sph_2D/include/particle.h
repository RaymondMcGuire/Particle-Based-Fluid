#ifndef _Particle_H
#define _Particle_H

#include "vector2.h"

class Particle
{
public:
    Vector2<float> position;
    Vector2<float> velocity;
    Vector2<float> force;

    float mass;
    float density;
    float pressure;

    float color;
    Vector2<float> normal;

    Particle();
    Particle(Vector2<float> position);

    float getVelocityLength2() const;
    float getForceLength2() const;
    float getNormalLength2() const;
};
#endif