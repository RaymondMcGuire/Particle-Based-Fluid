/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:40:11
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\shadow\point_shadow.h
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

    void enable(Vector3F) override;
    void bind() override;
    void release() override;

    KiriMaterialPtr getShadowDepthMaterial() override;

    UInt getDepthCubeMap();

    Vector3F pointLight;
    float near_plane = 1.0f;
    float mFarPlane = 25.0f;

private:
    UInt depthCubeMap;
    KiriMaterialPointShadowDepthPtr pointShadowDepthMaterial;
};

#endif