/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 01:48:41
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\mesh\mesh_internal.h
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
    void UpdateInstance(Array1Mat4x4F);
    virtual ~KiriMeshInternal(){};

protected:
    bool static_mesh = true;

    virtual void Construct() = 0;
    UInt mVBO, mVAO, mEBO, instanceVBO;

    // indices
    bool drawElem;
    Array1<UInt> indices;

    // vertices
    size_t verticesNum;
    DataType vertDataType;
    Array1<VertexStandard> vertStand;
    Array1<VertexFull> vertFull;
    Array1<VertexQuad2> vertQuad2;
    Array1<VertexQuad3> vertQuad3;
    void SetupVertex();
    void addVertStand(Vector3F, Vector3F, Vector2F);

    // instance
    bool instancing;
    Int instanceVertNum;
    Int instanceType;
    Array1<Matrix4x4F> instMat4;
    Array1<Vector2F> instVec2;

private:
    const size_t MAX_INSTANCE_NUM = 100000;
    const Int POSITION_LENGTH = 3;
    const Int COLOR_LENGTH = 3;
    const Int NORMAL_LENGTH = 3;
    const Int TEXCOORD_LENGTH = 2;
    const Int TANGENT_LENGTH = 3;
    const Int BITANGENT_LENGTH = 3;

    void initInstanceVBO(Int);
    void initInstanceMat4(Array1<Matrix4x4F>);
    void initInstanceVec2(Array1<Vector2F>);
    void SetupInstanceMat4(Int);
    void SetupInstanceVec2(Int);

    Array1<InstanceMat4x4> cvtMat4ToFloatArray(Array1<Matrix4x4F> mat4);
};
#endif
