/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:34:37
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\mesh\mesh_sphere.h
 */

#ifndef _KIRI_MESH_SPHERE_H_
#define _KIRI_MESH_SPHERE_H_
#pragma once
#include <kiri_core/mesh/mesh_internal.h>

class KiriMeshSphere : public KiriMeshInternal
{
public:
    KiriMeshSphere();
    KiriMeshSphere(float);
    KiriMeshSphere(Array1<Matrix4x4F>, bool = true);
    KiriMeshSphere(float, Int, Int, Array1<Matrix4x4F>, bool = true);
    ~KiriMeshSphere(){};

    void Draw() override;

private:
    float radius = 1.0f;

    Int X_SEGMENTS = 64;
    Int Y_SEGMENTS = 64;

    void Construct() override;
};
#endif
