#ifndef _SPH_Solver_H
#define _SPH_Solver_H

#include <vector>

#include "Particle.h"
#include "Grid.h"

class SPHSolver
{
public:
    SPHSolver();

    void update(float dt);

private:
    int numberParticles;
    std::vector<Particle> particles;
    std::vector<std::vector<int>> neighborhoods;
    Grid grid;

    float kernel(Vector2<float>, float h);
    Vector2<float> gradKernel(Vector2<float>, float h);
    float laplaceKernel(Vector2<float>, float h);

    void findNeighborhoods();

    void calculateDensity();
    void calculatePressure();

    void calculateForceDensity();

    void integrationStep(float dt);

    void collisionHandling();
};

#endif