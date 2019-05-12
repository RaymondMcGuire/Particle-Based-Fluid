#include"pbs_obj.h"

PbsObj::PbsObj(){}

PbsObj::~PbsObj(){}

inline string Vec2FToStr(const Vector2<float> v)
{
	stringstream ss;
	ss << v[0] << " " << v[1] ;
	return ss.str();
}

inline string Vec3FToStr(const Vector3<float> v)
{
	stringstream ss;
	ss << v[0] << " " << v[1] << " " << v[2];
	return ss.str();
}

inline string FaceVTNToStr(const Vector3<unsigned int> v, const Vector3<unsigned int> t, const Vector3<unsigned int> n)
{
	stringstream ss;
	ss << v[0] << "/" << t[0] << "/" << n[0] << " " << v[1] << "/" << t[1] << "/" << n[1] << " " << v[2] << "/" << t[2] << "/" << n[2];
	return ss.str();
}

inline string FaceVNToStr(const Vector3<unsigned int> v, const Vector3<unsigned int> n)
{
	stringstream ss;
	ss << v[0] << "//" << n[0] << " " << v[1] << "//" << n[1] << " " << v[2] << "//" << n[2];
	return ss.str();
}

bool PbsObj::Save(string file_name, 
                    const vector<Vector3<float>> &verts, 
                    const vector<Vector3<float>> &normals, 
                    const PbsFaceVTN &indices)
{

	ofstream file;

	file.open(file_name.c_str());
	if(!file || !file.is_open() || file.bad() || file.fail()){
		cout << "rxOBJ::Save : Invalid file specified" << endl;
		return false;
	}

	// write vertices data
	int nv = (int)verts.size();
	for(int i = 0; i < nv; ++i){
		file << "v " << Vec3FToStr(Vector3<float>(verts[i])) << endl;
	}

	// write coords data (temp)
	
	file << "vt " << Vec2FToStr(Vector2<float>(0.0f,0.0f)) << endl;
	file << "vt " << Vec2FToStr(Vector2<float>(1.0f,0.0f)) << endl;
	file << "vt " << Vec2FToStr(Vector2<float>(1.0f,1.0f)) << endl;

	// write normals data
	int nn = (int)normals.size();
	for(int i = 0; i < nn; ++i){
		file << "vn " << Vec3FToStr(Vector3<float>(normals[i])) << endl;
	}

	// write indices data
    int ni = (int)indices.n_indices.size();
	for(int i = 0; i < ni; ++i){
		file << "f " << FaceVTNToStr(Vector3<unsigned int>(indices.v_indices[i]),Vector3<unsigned int>(1,2,3) ,Vector3<unsigned int>(indices.n_indices[i])) << endl;
	}

	file.close();

	return true;
}

bool PbsObj::Save(string file_name, 
                    const vector<Vector3<float>> &verts, 
                    const vector<Vector3<float>> &normals, 
                    const PbsFaceVN &indices)
{

	ofstream file;

	file.open(file_name.c_str());
	if(!file || !file.is_open() || file.bad() || file.fail()){
		cout << "Please confirm your use the right path" << endl;
		return false;
	}

	// write vertices data
	int nv = (int)verts.size();
	for(int i = 0; i < nv; ++i){
		file << "v " << Vec3FToStr(Vector3<float>(verts[i])) << endl;
	}

	// write normals data
	int nn = (int)normals.size();
	for(int i = 0; i < nn; ++i){
		file << "vn " << Vec3FToStr(Vector3<float>(normals[i])) << endl;
	}

	// write indices data
    int ni = (int)indices.n_indices.size();
	for(int i = 0; i < ni; ++i){
		file << "f " << FaceVNToStr(Vector3<unsigned int>(indices.v_indices[i]),Vector3<unsigned int>(indices.n_indices[i])) << endl;
	}

	file.close();

	return true;
}