#ifndef _SPH_Solver_H
#define _SPH_Solver_H

#include <vector>

#include"json.hpp"
#include "particle.h"
#include "grid.h"

class SPHSolver
{
public:
	nlohmann::json data;
	int numberParticles;
	std::vector<Particle> particles;

	float currentTime;

    SPHSolver();
    void update(float dt);

private:
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
	void record(float dt);

    void collisionHandling();
};

#endif