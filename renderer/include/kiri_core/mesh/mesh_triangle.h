/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:49:58
 * @FilePath: \core\include\kiri_core\mesh\mesh_triangle.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
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

    Array1Vec3F GetVertices() const { return mVertices; };
    Array1Vec3F GetNormals() const { return mNormals; };
    Array1Vec3F GetTriangles() const { return mTriangles; };

    void GetEdges(Array1<std::pair<Int, Int>> &_edges) const;

private:
    Array1Vec3F mVertices;
    Array1Vec3F mNormals;
    Array1Vec3F mTriangles;
};

typedef SharedPtr<KiriMeshTriangle> KiriMeshTrianglePtr;

#endif
