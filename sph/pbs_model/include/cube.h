#ifndef _CUBE_H
#define _CUBE_H

#include <vector>
#include"vector3.h"
using namespace std;
class Cube
{
public:

    Cube(){set(Vector3<float>(),1.0);};
    Cube(Vector3<float> cen, float el){set(cen,el);};
    ~Cube() {}

    void set(Vector3<float> cen, float el);
    float getEdgeLength() const{ return edge_length; }
    Vector3<float> getCenter() const{ return center; }

    const vector<Vector3<float>> getVertices() const{ return vertices; }
    const vector<Vector3<float>> getNormals() const{ return normals; }
    const vector<Vector3<unsigned int>> getVertIndices() const{ return v_indices; }
    const vector<Vector3<unsigned int>> getNormIndices() const{ return n_indices; }

private:

    float edge_length;
    Vector3<float> center;

    vector<Vector3<float>> vertices;
    vector<Vector3<float>> normals;
    vector<Vector3<unsigned int>> v_indices;
    vector<Vector3<unsigned int>> n_indices;
    void clearArrays();
    void buildVertices();
};

#endif
