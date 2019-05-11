#ifndef _PBS_OBJ_H
#define _PBS_OBJ_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include"vector2.h"
#include"vector3.h"

using namespace std;

struct PbsFaceVN{
    vector<Vector3<unsigned int>> v_indices;
    vector<Vector3<unsigned int>> n_indices;
};

struct PbsFaceVTN{
    vector<Vector3<unsigned int>> v_indices;
    vector<Vector3<unsigned int>> n_indices;
};

class PbsObj
{
public:
	PbsObj();
	~PbsObj();

	bool Save(string file_name, const vector<Vector3<float>> &verts, const vector<Vector3<float>> &normals, const PbsFaceVN &indices);
	bool Save(string file_name, const vector<Vector3<float>> &verts, const vector<Vector3<float>> &normals, const PbsFaceVTN &indices);
};

#endif