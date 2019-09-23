//#include "sph3_data.h"
//
//void SPH3Data::initSPHData(){
//    Vector3<float> start = worldSize*0.2f;
//    Vector3<float> end = worldSize*0.8f;
//
//    float interval = kernel*0.5f;
//
//    int numX = int((end.x-start.x)/interval);
//    int numY = int((end.y-start.y)/interval);
//    int numZ = int((end.z-start.z)/interval);
//
//    for (int i = 0; i < numX; i++)
//    {
//        for (int j = 0; j < numY; j++)
//        {
//            for (int k = 0; k < numZ; k++)
//            {
//                const Vector3<float> position =  start + Vector3<float>(float(i),float(j),float(k)) * interval;
//                Particle3 p = Particle3(position);
//                addParticle(p);
//            }
//        }
//    }
//
//    cout<<"particle number:"<<numberParticles<<endl;
//    
//}
//
//void SPH3Data::addParticle(Particle3& p){
//    particles.push_back(p);
//    numberParticles++;
//}