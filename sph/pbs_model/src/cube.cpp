#include"cube.h"

void Cube::clearArrays()
{
    std::vector<Vector3<float>>().swap(vertices);
    std::vector<Vector3<float>>().swap(normals);
    std::vector<Vector3<unsigned int>>().swap(v_indices);
    std::vector<Vector3<unsigned int>>().swap(n_indices);
}


void Cube::set(Vector3<float> cen, float el)
{
    this->center = cen;
    this->edge_length = el;

    clearArrays();
    buildVertices();
}


void Cube::buildVertices(){
    float h_el = this->edge_length/2;
    //  5-----7
    // /|    /|
    //3-----1 |
    //| 6---|-8
    //|/    |/
    //4-----2
    //vertices
    this->vertices.push_back(Vector3<float>(this->center.x+h_el,this->center.y+h_el,this->center.z+h_el));
    this->vertices.push_back(Vector3<float>(this->center.x+h_el,this->center.y+h_el,this->center.z-h_el));
    this->vertices.push_back(Vector3<float>(this->center.x-h_el,this->center.y+h_el,this->center.z+h_el));
    this->vertices.push_back(Vector3<float>(this->center.x-h_el,this->center.y+h_el,this->center.z-h_el));
    this->vertices.push_back(Vector3<float>(this->center.x-h_el,this->center.y-h_el,this->center.z+h_el));
    this->vertices.push_back(Vector3<float>(this->center.x-h_el,this->center.y-h_el,this->center.z-h_el));
    this->vertices.push_back(Vector3<float>(this->center.x+h_el,this->center.y-h_el,this->center.z+h_el));
    this->vertices.push_back(Vector3<float>(this->center.x+h_el,this->center.y-h_el,this->center.z-h_el));
    //normals
    this->normals.push_back(Vector3<float>( 0.0f, 1.0f, 0.0f));
    this->normals.push_back(Vector3<float>(-1.0f, 0.0f, 0.0f));
    this->normals.push_back(Vector3<float>( 0.0f,-1.0f, 0.0f));
    this->normals.push_back(Vector3<float>( 1.0f, 0.0f, 0.0f));
    this->normals.push_back(Vector3<float>( 0.0f, 0.0f, 1.0f));
    this->normals.push_back(Vector3<float>( 0.0f, 0.0f,-1.0f));
    //vertices indices
    this->v_indices.push_back(Vector3<unsigned int>(1,2,4));
    this->v_indices.push_back(Vector3<unsigned int>(1,4,3));

    this->v_indices.push_back(Vector3<unsigned int>(3,4,6));
    this->v_indices.push_back(Vector3<unsigned int>(3,6,5));

    this->v_indices.push_back(Vector3<unsigned int>(5,6,8));
    this->v_indices.push_back(Vector3<unsigned int>(5,8,7));

    this->v_indices.push_back(Vector3<unsigned int>(1,7,8));
    this->v_indices.push_back(Vector3<unsigned int>(1,8,2));

    this->v_indices.push_back(Vector3<unsigned int>(5,7,1));
    this->v_indices.push_back(Vector3<unsigned int>(5,1,3));

    this->v_indices.push_back(Vector3<unsigned int>(6,2,8));
    this->v_indices.push_back(Vector3<unsigned int>(6,4,2));
    //normals indices
    this->n_indices.push_back(Vector3<unsigned int>(1,1,1));
    this->n_indices.push_back(Vector3<unsigned int>(1,1,1));

    this->n_indices.push_back(Vector3<unsigned int>(2,2,2));
    this->n_indices.push_back(Vector3<unsigned int>(2,2,2));

    this->n_indices.push_back(Vector3<unsigned int>(3,3,3));
    this->n_indices.push_back(Vector3<unsigned int>(3,3,3));

    this->n_indices.push_back(Vector3<unsigned int>(4,4,4));
    this->n_indices.push_back(Vector3<unsigned int>(4,4,4));

    this->n_indices.push_back(Vector3<unsigned int>(5,5,5));
    this->n_indices.push_back(Vector3<unsigned int>(5,5,5));

    this->n_indices.push_back(Vector3<unsigned int>(6,6,6));
    this->n_indices.push_back(Vector3<unsigned int>(6,6,6));

}