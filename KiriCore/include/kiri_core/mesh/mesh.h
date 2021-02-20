/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-20 20:51:05
 * @LastEditTime: 2021-02-20 19:35:06
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\mesh\mesh.h
 */

#ifndef _KIRI_MESH_H_
#define _KIRI_MESH_H_

#pragma once

#include <kiri_core/kiri_shader.h>

class KiriMesh
{
public:
    KiriMesh();
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
    Array1<VertexStandard> vertStand;
    Array1<VertexFull> vertFull;

    Array1<UInt> indices;
    Array1<Texture> textures;
    UInt mVAO, mVBO, mEBO, instanceVBO;

    DataType type;
    void SetupStand();
    void SetupFull();

    //instance
    bool instancing = false;
    Array1<Matrix4x4F> trans4;
    void SetupInstance(Int);
    void instMat4(Array1<Matrix4x4F>);
};
#endif
