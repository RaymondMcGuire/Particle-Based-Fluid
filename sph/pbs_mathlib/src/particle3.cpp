#include "particle3.h"
#include "constants.h"
#include "vector3.h"

Particle3::Particle3(){

}

Particle3::Particle3(Vector3<float> pos)
{
    position = pos;
    velocity = Vector3<float>();
    force = Vector3<float>();

    mass = Constants::PARTICLE_MASS;

    density = 0;
    pressure = 0;

    color = 0;
    normal = Vector3<float>();
}

float Particle3::getVelocityLength2() const
{
    return velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z;
}

float Particle3::getForceLength2() const
{
    return force.x * force.x + force.y * force.y + force.z * force.z;
}

float Particle3::getNormalLength2() const
{
    return normal.x * normal.x + normal.y * normal.y+ normal.z * normal.z;
}
