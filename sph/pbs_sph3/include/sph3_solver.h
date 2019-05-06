#ifndef _SPH3_Solver_H
#define _SPH3_Solver_H
#include <vector>
#include "particle3.h"
#include "sph3_data.h"

class SPH3Solver
{
public:

	float currentTime;

    SPH3Solver();

private:
	SPH3Data* sph3_data;
};

#endif