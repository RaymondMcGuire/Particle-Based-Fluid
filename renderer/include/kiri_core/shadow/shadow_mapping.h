/***
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:40:06
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \KiriCore\include\kiri_core\mShadow\shadow_mapping.h
 */

#ifndef _KIRI_SHADOW_MAPPING_H_
#define _KIRI_SHADOW_MAPPING_H_
#pragma once
#include <kiri_core/shadow/shadow.h>
#include <kiri_core/material/material_shadow_depth.h>

class KiriShadowMapping : public KiriShadow
{
public:
    KiriShadowMapping();

    void Enable(Vector3F) override;
    void Bind() override;
    void Release() override;

    KiriMaterialPtr GetShadowDepthMaterial() override;

    Matrix4x4F getLightSpaceMatrix();
    UInt getDepthMap();

private:
    UInt depthMap;
    Vector3F directionLight;
    Matrix4x4F mLightSpaceMatrix;

    KiriMaterialShadowDepthPtr shadowDepthMaterial;
};

#endif