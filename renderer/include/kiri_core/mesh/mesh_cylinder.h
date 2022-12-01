/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:39:26
 * @FilePath: \core\include\kiri_core\mesh\mesh_cylinder.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _KIRI_MESH_CYLINDER_H_
#define _KIRI_MESH_CYLINDER_H_
#pragma once
#include <kiri_core/mesh/mesh_internal.h>

class KiriMeshCylinder : public KiriMeshInternal
{
public:
    KiriMeshCylinder(float baseRadius = 1.0f, float topRadius = 1.0f, float height = 2.0f,
                     Int sectorCount = 36, Int stackCount = 8, bool smooth = true);
    ~KiriMeshCylinder(){};

    void Draw() override;

private:
    // constants
    const Int MIN_SECTOR_COUNT = 3;
    const Int MIN_STACK_COUNT = 1;

    // params
    float mBaseRadius;
    float mTopRadius;
    float mHeight;
    Int mSectorCount; // slices
    Int mStackCount;  // stacks
    UInt mBaseIndex;  // starting index of base
    UInt mTopIndex;   // starting index of top
    bool mSmooth;

    Array1<float> mUnitCircleVertices;
    void BuildUnitCircleVertices();
    void BuildVerticesSmooth();

    Array1<UInt> lineIndices;
    void Construct() override;

    Array1<float> GetSideNormals();
};
#endif
