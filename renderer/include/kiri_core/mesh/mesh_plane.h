/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:44:56
 * @FilePath: \core\include\kiri_core\mesh\mesh_plane.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
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
    float GetWidth()
    {
        return mWidth;
    }

    float GetY()
    {
        return y;
    }

    Vector3F GetNormal()
    {
        return mNormal;
    }

private:
    float mWidth = 10.0f;
    float y = -0.5f;
    Vector3F mNormal = Vector3F(0.0f, 1.0f, 0.0f);
    void Construct() override;

    Array1<float> mPlaneVertices = {
        // positions            // normals         // texcoords
        mWidth, y, mWidth, mNormal.x, mNormal.y, mNormal.z, mWidth, 0.0f,
        -mWidth, y, mWidth, mNormal.x, mNormal.y, mNormal.z, 0.0f, 0.0f,
        -mWidth, y, -mWidth, mNormal.x, mNormal.y, mNormal.z, 0.0f, mWidth,

        mWidth, y, mWidth, mNormal.x, mNormal.y, mNormal.z, mWidth, 0.0f,
        -mWidth, y, -mWidth, mNormal.x, mNormal.y, mNormal.z, 0.0f, mWidth,
        mWidth, y, -mWidth, mNormal.x, mNormal.y, mNormal.z, mWidth, mWidth};
};
#endif
