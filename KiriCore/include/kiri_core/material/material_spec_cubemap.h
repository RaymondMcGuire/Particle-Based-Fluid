/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:04:26
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_spec_cubemap.h
 */

#ifndef _KIRI_MATERIAL_SPEC_CUBEMAP_H_
#define _KIRI_MATERIAL_SPEC_CUBEMAP_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialSpecCubeMap : public KiriMaterial
{
public:
    KiriMaterialSpecCubeMap(Matrix4x4F);

    void Setup() override;
    void Update() override;

private:
    Matrix4x4F mCaptureProjection;
};
typedef SharedPtr<KiriMaterialSpecCubeMap> KiriMaterialSpecCubeMapPtr;
#endif