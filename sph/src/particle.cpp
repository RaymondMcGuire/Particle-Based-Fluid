#include "particle.h"
#include "constants.h"
#include "vector2.h"

Particle::Particle()
{
    Particle(Vector2f());
}

Particle::Particle(Vector2<float> pos)
{
    position = pos;
    velocity = Vector2<float>();
    force = Vector2<float>();

    mass = Constants::PARTICLE_MASS;

    density = 0;
    pressure = 0;

    color = 0;
    normal = Vector2<float>();
}

float Particle::getVelocityLength2() const
{
    return velocity.x * velocity.x + velocity.y * velocity.y;
}

float Particle::getForceLength2() const
{
    return force.x * force.x + force.y * force.y;
}

float Particle::getNormalLength2() const
{
    return normal.x * normal.x + normal.y * normal.y;
}
