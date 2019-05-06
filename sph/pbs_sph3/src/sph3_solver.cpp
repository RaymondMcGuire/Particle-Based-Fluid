#include <iostream>
#include <algorithm>

#include "sph3_solver.h"
#include "constants.h"

using namespace std;
using namespace Constants;


SPH3Solver::SPH3Solver()
{
    cout<<"construct sph3 solver!"<<endl;
    sph3_data = new SPH3Data();
    sph3_data->initSPHData();
    
    // for each(const Particle3 &p in sph3_data->particles){
    //     cout<<"particle position:("<<p.position.x<<","<<p.position.y<<","<<p.position.z<<")"<<endl;
    // }
}