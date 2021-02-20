/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 18:46:42
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_shadow_depth.h
 */

#ifndef _KIRI_MATERIAL_SHADOW_DEPTH_H_
#define _KIRI_MATERIAL_SHADOW_DEPTH_H_

#pragma once

#include <kiri_core/material/material.h>

class KiriMaterialShadowDepth : public KiriMaterial
{
public:
    KiriMaterialShadowDepth();

    void Setup() override;
    void Update() override;

    void SetLightSpaceMatrix(Matrix4x4F);

private:
    Matrix4x4F mLightSpaceMatrix;
};
typedef SharedPtr<KiriMaterialShadowDepth> KiriMaterialShadowDepthPtr;
#endif