/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:48:59
 * @FilePath: \core\include\kiri_core\mesh\mesh_sphere.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
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
    float mRadius = 1.0f;

    Int X_SEGMENTS = 64;
    Int Y_SEGMENTS = 64;

    void Construct() override;
};
#endif
