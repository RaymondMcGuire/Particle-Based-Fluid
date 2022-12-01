/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:38:02
 * @FilePath: \core\include\kiri_core\mesh\mesh_pbr.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
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
    Array1<VertexStandard> mVertStand;
    Array1<VertexFull> mVertFull;

    Array1<UInt> mIndices;
    Array1<Texture> mTextures;
    UInt mVAO, mVBO, mEBO;

    DataType mDataType;
    void SetupStand();
    void SetupFull();
};

typedef SharedPtr<KiriMeshPBR> KiriMeshPBRPtr;
#endif
