/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:32:09
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\mesh\mesh_cylinder.h
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
    float baseRadius;
    float topRadius;
    float height;
    Int sectorCount; // slices
    Int stackCount;  // stacks
    UInt baseIndex;  // starting index of base
    UInt topIndex;   // starting index of top
    bool smooth;

    Array1<float> unitCircleVertices;
    void buildUnitCircleVertices();
    void buildVerticesSmooth();

    Array1<UInt> lineIndices;
    void Construct() override;

    Array1<float> getSideNormals();
};
#endif
