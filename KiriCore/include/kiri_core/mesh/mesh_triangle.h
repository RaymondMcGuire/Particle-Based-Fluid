/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-19 18:18:39
 * @LastEditTime: 2021-02-20 19:34:50
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\mesh\mesh_triangle.h
 */

#ifndef _KIRI_MESH_TRIANGLE_H_
#define _KIRI_MESH_TRIANGLE_H_
#pragma once
#include <kiri_pch.h>

class KiriMeshTriangle
{
public:
    KiriMeshTriangle();
    KiriMeshTriangle(Array1Vec3F vertices, Array1Vec3F normals, Array1Vec3F triangles);
    ~KiriMeshTriangle();

    Array1Vec3F vertices() const { return mVertices; };
    Array1Vec3F normals() const { return mNormals; };
    Array1Vec3F triangles() const { return mTriangles; };

    void GetEdges(Array1<std::pair<Int, Int>> &_edges) const;

private:
    Array1Vec3F mVertices;
    Array1Vec3F mNormals;
    Array1Vec3F mTriangles;
};

typedef SharedPtr<KiriMeshTriangle> KiriMeshTrianglePtr;

#endif
