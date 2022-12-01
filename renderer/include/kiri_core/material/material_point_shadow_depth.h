/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:10:39
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material_point_shadow_depth.h
 */

#ifndef _KIRI_MATERIAL_POINT_SHADOW_DEPTH_H_
#define _KIRI_MATERIAL_POINT_SHADOW_DEPTH_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriMaterialPointShadowDepth : public KiriMaterial
{
public:
    KiriMaterialPointShadowDepth();

    void Setup() override;
    void Update() override;

    void SetShadowTransforms(Array1<Matrix4x4F>);

    void SetFarPlane(float);
    void SetLightPos(Vector3F);

private:
    Array1<Matrix4x4F> mShadowTransforms;
    float mFarPlane;
    Vector3F mLightPos;
};
typedef SharedPtr<KiriMaterialPointShadowDepth> KiriMaterialPointShadowDepthPtr;
#endif