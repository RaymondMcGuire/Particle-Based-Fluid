//#include <iostream>
//#include <algorithm>
//
//#include "sph3_solver.h"
//#include "constants.h"
//
//using namespace std;
//using namespace Constants;
//
//
//SPH3Solver::SPH3Solver(){
//    cout<<"construct sph3 solver!"<<endl;
//    sph3_data = new SPH3Data();
//    sph3_data->initSPHData();
//
//}
//
//vector<Vector3<float>> SPH3Solver::GetParticlesPos(){
//    vector<Vector3<float>> particlesPos;
//    for each(const Particle3 &p in sph3_data->particles){
//        //cout<<"particle position:("<<p.position.x<<","<<p.position.y<<","<<p.position.z<<")"<<endl;
//        particlesPos.push_back(Vector3<float>(p.position.x,p.position.y,p.position.z));
//    }
//    return particlesPos;
//}