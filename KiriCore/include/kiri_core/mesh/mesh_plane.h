/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 19:32:26
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\mesh\mesh_plane.h
 */

#ifndef _KIRI_MESH_PLANE_H_
#define _KIRI_MESH_PLANE_H_
#pragma once
#include <kiri_core/mesh/mesh_internal.h>

class KiriMeshPlane : public KiriMeshInternal
{
public:
    KiriMeshPlane();
    KiriMeshPlane(float, float, Vector3F);
    ~KiriMeshPlane(){};

    void Draw() override;
    float getWidth()
    {
        return width;
    }

    float getY()
    {
        return y;
    }

    Vector3F getNormal()
    {
        return normal;
    }

private:
    float width = 10.0f;
    float y = -0.5f;
    Vector3F normal = Vector3F(0.0f, 1.0f, 0.0f);
    void Construct() override;

    Array1<float> planeVertices = {
        // positions            // normals         // texcoords
        width, y, width, normal.x, normal.y, normal.z, width, 0.0f,
        -width, y, width, normal.x, normal.y, normal.z, 0.0f, 0.0f,
        -width, y, -width, normal.x, normal.y, normal.z, 0.0f, width,

        width, y, width, normal.x, normal.y, normal.z, width, 0.0f,
        -width, y, -width, normal.x, normal.y, normal.z, 0.0f, width,
        width, y, -width, normal.x, normal.y, normal.z, width, width};
};
#endif
