/***
 * @Author: Xu.WANG
 * @Date: 2020-10-20 20:51:05
 * @LastEditTime: 2021-04-07 14:01:47
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriCore\include\kiri_core\mesh\mesh.h
 */

#ifndef _KIRI_MESH_H_
#define _KIRI_MESH_H_

#pragma once

#include <kiri_core/kiri_shader.h>

class KiriMesh
{
public:
    KiriMesh();
    KiriMesh(Array1<VertexSimple>);
    KiriMesh(Array1<VertexStandard>, Array1<UInt>, bool, Array1<Matrix4x4F>);
    KiriMesh(Array1<VertexFull>, Array1<UInt>, bool, Array1<Matrix4x4F>);
    KiriMesh(Array1<VertexStandard>, Array1<UInt>, Array1<Texture>, bool, Array1<Matrix4x4F>);
    KiriMesh(Array1<VertexFull>, Array1<UInt>, Array1<Texture>, bool, Array1<Matrix4x4F>);
    ~KiriMesh();

    void Draw(KiriShader *);

    UInt GetVAO()
    {
        return mVAO;
    }

    UInt GetVBO()
    {
        return mVBO;
    }

    UInt GetEBO()
    {
        return mEBO;
    }

private:
    Array1<VertexSimple> mVertSimple;
    Array1<VertexStandard> mVertStand;
    Array1<VertexFull> mVertFull;

    Array1<UInt> mIndices;
    Array1<Texture> mTextures;
    UInt mVAO, mVBO, mEBO, instanceVBO;

    DataType type;
    void SetupSimple();
    void SetupStand();
    void SetupFull();

    // instance
    bool mInstance = false;
    Array1<Matrix4x4F> trans4;
    void SetupInstance(Int);
    void mInstMat4(Array1<Matrix4x4F>);
};
#endif
