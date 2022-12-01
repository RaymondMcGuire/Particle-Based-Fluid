/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:47:50
 * @FilePath: \core\include\kiri_core\mesh\mesh_internal.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _KIRI_MESH_INTERNAL_H_
#define _KIRI_MESH_INTERNAL_H_

#pragma once

#include <kiri_pch.h>
#include <glad/glad.h>
class KiriMeshInternal
{
public:
    virtual void Draw() = 0;
    void UpdateInstance(Array1Mat4x4F instMat4);
    virtual ~KiriMeshInternal(){};

protected:
    bool mStaticMesh = true;

    virtual void Construct() = 0;
    UInt mVBO, mVAO, mEBO, mInstanceVBO;

    // mIndices
    bool mDrawElem;
    Array1<UInt> mIndices;

    // vertices
    size_t mVerticesNum;
    DataType mVertDataType;
    Array1<VertexSimple> mVertSimple;
    Array1<VertexStandard> mVertStand;
    Array1<VertexFull> mVertFull;
    Array1<VertexQuad2> mVertQuad2;
    Array1<VertexQuad3> mVertQuad3;
    void SetupVertex();
    void AddVertSimple(Vector3F pos, Vector3F col);
    void AddVertStand(Vector3F pos, Vector3F norm, Vector2F tex);

    // instance
    bool mInstance;
    Int mInstanceVertNum;
    Int mInstanceType;
    Array1<Matrix4x4F> mInstMat4;
    Array1<Vector2F> mInstVec2;

private:
    const size_t MAX_INSTANCE_NUM = 100000;
    const Int POSITION_LENGTH = 3;
    const Int COLOR_LENGTH = 3;
    const Int NORMAL_LENGTH = 3;
    const Int TEXCOORD_LENGTH = 2;
    const Int TANGENT_LENGTH = 3;
    const Int BITANGENT_LENGTH = 3;

    void InitInstanceVBO(Int);
    void InitInstanceMat4(Array1<Matrix4x4F> instMat4);
    void InitInstanceVec2(Array1<Vector2F> instVec2);
    void SetupInstanceMat4(Int);
    void SetupInstanceVec2(Int);

    Array1<InstanceMat4x4> ConvertMat4ToFloatArray(Array1<Matrix4x4F> instMat4);
};
#endif
