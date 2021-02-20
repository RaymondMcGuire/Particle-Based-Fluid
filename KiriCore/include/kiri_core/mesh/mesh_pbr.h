/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 19:34:23
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\mesh\mesh_pbr.h
 */

#ifndef _KIRI_MESH_PBR_H_
#define _KIRI_MESH_PBR_H_

#pragma once

#include <kiri_pch.h>

class KiriMeshPBR
{
public:
    KiriMeshPBR();
    KiriMeshPBR(Array1<VertexStandard>, Array1<UInt>);
    KiriMeshPBR(Array1<VertexFull>, Array1<UInt>);
    KiriMeshPBR(Array1<VertexStandard>, Array1<UInt>, Array1<Texture>);
    KiriMeshPBR(Array1<VertexFull>, Array1<UInt>, Array1<Texture>);
    ~KiriMeshPBR();

    void Draw();

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
    UInt mVAO, mVBO, mEBO;

    DataType type;
    void SetupStand();
    void SetupFull();
};

typedef SharedPtr<KiriMeshPBR> KiriMeshPBRPtr;
#endif
