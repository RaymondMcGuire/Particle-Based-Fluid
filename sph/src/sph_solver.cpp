#include "sph_solver.h"

#include <iostream>
#include <algorithm>

#include "constants.h"

using namespace std;
using namespace Constants;

struct Rect
{
    float left;
    float top;
    float width;
    float height;
};

SPHSolver::SPHSolver()
{
	currentTime = 0.0f;
	//data["frames"] = nlohmann::json::array();

    int particlesX = NUMBER_PARTICLES / 2.0f;
    int particlesY = NUMBER_PARTICLES;

    numberParticles = particlesX * particlesY;
    particles = vector<Particle>();

    float width = WIDTH / 4.2f;
    float height = 3.0f * HEIGHT / 4.0f;

    Rect particleRect = Rect{ (WIDTH - width) / 2, HEIGHT - height, width, height};

	cout << "Particle Radius:" << 0.5f * PARTICLE_SPACING * SCALE << endl;
	cout << "Rect Width:" << RENDER_WIDTH << ", Height:" << RENDER_HEIGHT << endl;

    float dx = particleRect.width / particlesX;
    float dy = particleRect.height / particlesY;

    for (int i = 0; i < NUMBER_PARTICLES / 2.0f; i++)
    {
        for (int j = 0; j < NUMBER_PARTICLES; j++)
        {
            Vector2<float> pos = Vector2<float>(particleRect.left, particleRect.top) + Vector2<float>(i * dx, j * dy);
			
            Particle p = Particle(pos);
            particles.push_back(p);
        }
    }

    grid.updateStructure(particles);

    cout << "SPH Solver initialized with " << numberParticles << " particles." << endl;
}

void SPHSolver::update(float dt)
{
	//record(dt);
    findNeighborhoods();
    calculateDensity();
    calculatePressure();
    calculateForceDensity();
    integrationStep(dt);
    collisionHandling();
    grid.updateStructure(particles);
}

// Poly6 Kernel
float SPHSolver::kernel(Vector2<float> x, float h)
{
    float r2 = x.x * x.x + x.y * x.y;
    float h2 = h * h;

    if (r2 < 0 || r2 > h2)
        return 0.0f;

    return 315.0f / (64.0f * M_PI * pow(h, 9)) * pow(h2 - r2, 3);
}

// Gradient of Spiky Kernel
Vector2<float> SPHSolver::gradKernel(Vector2<float> x, float h)
{
    float r = sqrt(x.x * x.x + x.y * x.y);
    if (r == 0.0f)
        return Vector2<float>(0.0f, 0.0f);

    float t1 = -45.0f / (M_PI * pow(h, 6));
    Vector2<float> t2 = x / r;
    float t3 = pow(h - r, 2);

    return t2 * t1 * t3;
}

// Laplacian of Viscosity Kernel
float SPHSolver::laplaceKernel(Vector2<float> x, float h)
{
    float r = sqrt(x.x * x.x + x.y * x.y);
    return 45.0f / (M_PI * pow(h, 6)) * (h - r);
}

void SPHSolver::findNeighborhoods()
{
    neighborhoods = vector<vector<int>>();
    float maxDist2 = KERNEL_RANGE * KERNEL_RANGE;

    for each(const Particle &p in particles)
        {
            vector<int> neighbors = vector<int>();
            vector<Cell> neighboringCells = grid.getNeighboringCells(p.position);

        for each(const Cell &cell in neighboringCells)
            {
            for each(int index in cell)
                {
                    Vector2<float> x = p.position - particles[index].position;
                    float dist2 = x.x * x.x + x.y * x.y;
                    if (dist2 <= maxDist2)
                    {
                        neighbors.push_back(index);
                    }
                }
            }

        neighborhoods.push_back(neighbors);
        }
}

void SPHSolver::calculateDensity()
{
    for (int i = 0; i < numberParticles; i++)
    {
        vector<int> neighbors = neighborhoods[i];
        float densitySum = 0.0f;

        for (int n = 0; n < neighbors.size(); n++)
        {
            int j = neighbors[n];

            Vector2<float> x = particles[i].position - particles[j].position;
            densitySum += particles[j].mass * kernel(x, KERNEL_RANGE);
        }

        particles[i].density = densitySum;
    }
}

void SPHSolver::calculatePressure()
{
    for (int i = 0; i < numberParticles; i++)
    {
        particles[i].pressure = max(STIFFNESS * (particles[i].density - REST_DENSITY), 0.0f);
    }
}

void SPHSolver::calculateForceDensity()
{
    for (int i = 0; i < numberParticles; i++)
    {
        Vector2<float> fPressure = Vector2<float>(0.0f, 0.0f);
        Vector2<float> fViscosity = Vector2<float>(0.0f, 0.0f);
        Vector2<float> fGravity = Vector2<float>(0.0f, 0.0f);

        vector<int> neighbors = neighborhoods[i];

        //particles[i].color = 0;

        for (int n = 0; n < neighbors.size(); n++)
        {
            int j = neighbors[n];
            Vector2<float> x = particles[i].position - particles[j].position;

            // Pressure force density
            fPressure += gradKernel(x, KERNEL_RANGE) * particles[j].mass * (particles[i].pressure + particles[j].pressure) / (2.0f * particles[j].density);

            // Viscosity force density
            fViscosity += (particles[j].velocity - particles[i].velocity) / particles[j].density * laplaceKernel(x, KERNEL_RANGE) * particles[j].mass;

            // Color field
            //particles[i].color += particles[j].mass / particles[j].density * kernel(x, KERNEL_RANGE);
        }

        // Gravitational force density
        fGravity = Vector2<float>(0, GRAVITY) * particles[i].density;

        fPressure *= -1.0f;
        fViscosity *= VISCOCITY;

        //particles[i].force += fPressure + fViscosity + fGravity + fSurface;
        particles[i].force += fPressure + fViscosity + fGravity;
    }
}

void SPHSolver::integrationStep(float dt)
{
    for (int i = 0; i < numberParticles; i++)
    {
        particles[i].velocity += particles[i].force / particles[i].density * dt;
        particles[i].position += particles[i].velocity * dt;
    }

}

void SPHSolver::record(float dt)
{
	nlohmann::json f;
	f["time"] = currentTime;
	f["particles"] = nlohmann::json::array();
	for (int i = 0; i < numberParticles; i++)
	{
		nlohmann::json p;
		//p["velocity"]["x"] = particles[i].velocity.x;
		//p["velocity"]["y"] = particles[i].velocity.y;
		p["position"]["x"] = particles[i].position.x * SCALE;
		p["position"]["y"] = particles[i].position.y * SCALE;
		//cout << particles[i].position.x << "_" << particles[i].position.y << endl;
		f["particles"].push_back(p);
	}
	data["frames"].push_back(f);

	cout<<"Recorded Frame:"<< currentTime << endl;
	currentTime += dt;
}

void SPHSolver::collisionHandling()
{
    for (int i = 0; i < numberParticles; i++)
    {
        if (particles[i].position.x < 0.0f)
        {
            particles[i].position.x = 0.0f;
            particles[i].velocity.x = -0.5f * particles[i].velocity.x;
        }
        else if (particles[i].position.x > WIDTH)
        {
            particles[i].position.x = WIDTH;
            particles[i].velocity.x = -0.5f * particles[i].velocity.x;
        }

        if (particles[i].position.y < 0.0f)
        {
            particles[i].position.y = 0.0f;
            particles[i].velocity.y = -0.5f * particles[i].velocity.y;
        }
        else if (particles[i].position.y > HEIGHT)
        {
            particles[i].position.y = HEIGHT;
            particles[i].velocity.y = -0.5f * particles[i].velocity.y;
        }
    }
}
