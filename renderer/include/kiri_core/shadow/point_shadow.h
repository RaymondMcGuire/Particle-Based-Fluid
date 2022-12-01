/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:25:12
 * @FilePath: \core\include\kiri_core\shadow\point_shadow.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _KIRI_POINT_SHADOW_H_
#define _KIRI_POINT_SHADOW_H_
#pragma once
#include <kiri_core/shadow/shadow.h>
#include <kiri_core/material/material_point_shadow_depth.h>

class KiriPointShadow : public KiriShadow
{
public:
    KiriPointShadow();

    void Enable(Vector3F) override;
    void Bind() override;
    void Release() override;

    KiriMaterialPtr GetShadowDepthMaterial() override;

    UInt GetDepthCubeMap();

    Vector3F mPointLight;
    float mNearPlane = 1.0f;
    float mFarPlane = 25.0f;

private:
    UInt mDepthCubeMapBuffer;
    KiriMaterialPointShadowDepthPtr mPointShadowDepthMat;
};

#endif