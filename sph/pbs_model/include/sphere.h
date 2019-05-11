#ifndef _SPHERE_H
#define _SPHERE_H

#include <vector>
#include"vector2.h"
#include"vector3.h"
using namespace std;
class Sphere
{
public:

    Sphere(){set(1.0f,36,18,true);};
    Sphere(float radius, int sectorCount, int stackCount, bool smooth){set(radius,sectorCount,stackCount,smooth);};
    ~Sphere() {}

    void set(float radius, int sectorCount, int stackCount, bool smooth=true);


    const vector<Vector3<float>> getVertices() const{ return vertices; }
    const vector<Vector3<float>> getNormals() const{ return normals; }
    const vector<Vector2<float>> getTexCoords() const{ return texcoords; }
    const vector<Vector3<unsigned int>> getVertIndices() const{ return v_indices; }
    const vector<Vector3<unsigned int>> getNormIndices() const{ return n_indices; }

private:

    const int MIN_SECTOR_COUNT = 3;
    const int MIN_STACK_COUNT  = 2;

    float radius;
    int sectorCount; // longitude
    int stackCount;  // latitude
    bool smooth;

    vector<Vector3<float>> vertices;
    vector<Vector3<float>> normals;
    vector<Vector2<float>> texcoords;
    vector<Vector3<unsigned int>> v_indices;
    vector<Vector3<unsigned int>> n_indices;
    void clearArrays();
    void buildVerticesSmooth();
    void buildVerticesFlat();
};

#endif
