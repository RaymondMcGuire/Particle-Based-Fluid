#include"pbs_obj.h"
#include"cube.h"
#include"sphere.h"

int main(){

    PbsObj pbs_obj = PbsObj();
    PbsFaceVN pbs_face_vn;
	PbsFaceVTN pbs_face_vtn;

	Cube cube = Cube();
    pbs_face_vn.n_indices = cube.getNormIndices();
    pbs_face_vn.v_indices = cube.getVertIndices();

    if(pbs_obj.Save("cube.obj",cube.getVertices(),cube.getNormals(),pbs_face_vn))
        cout<<"write cube.obj to file"<<endl;

	Sphere sphere = Sphere(0.01f, 36, 18, true);
    pbs_face_vtn.n_indices = sphere.getNormIndices();
    pbs_face_vtn.v_indices = sphere.getVertIndices();

    if(pbs_obj.Save("sphere.obj",sphere.getVertices(),sphere.getNormals(),pbs_face_vtn))
        cout<<"write sphere.obj to file"<<endl;

    return 0;
}