/***
 * @Author: Xu.WANG
 * @Date: 2021-12-23 16:39:32
 * @LastEditTime: 2021-12-23 16:40:03
 * @LastEditors: Xu.WANG
 * @Description:
 */

#ifndef _KIRI_MESH_CUSTOM_H_
#define _KIRI_MESH_CUSTOM_H_
#pragma once
#include <kiri_core/mesh/mesh_internal.h>

class KiriMeshCustom : public KiriMeshInternal
{
public:
    KiriMeshCustom();
    ~KiriMeshCustom(){};

    void Draw() override;
    void Generate(std::vector<Vector3D> pos, std::vector<Vector3D> mNormal, std::vector<int> mIndices);

private:
    void Construct() override;
};
#endif
