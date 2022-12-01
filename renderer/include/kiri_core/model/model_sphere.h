/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:34:51
 * @FilePath: \core\include\kiri_core\model\model_sphere.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */


#ifndef _KIRI_MODEL_SPHERE_H_
#define _KIRI_MODEL_SPHERE_H_
#pragma once
#include <kiri_core/model/model_internal.h>
#include <kiri_core/mesh/mesh_sphere.h>

class KiriSphere : public KiriModelInternal
{
public:
    KiriSphere();
    KiriSphere(float);
    KiriSphere(float, Int, Int, Array1<Matrix4x4F>, bool = true);
    KiriSphere(Array1<Matrix4x4F>, bool = true);

    void SetDiffuseMap(bool);
    void LoadDiffuseMap(UInt);
    void Draw() override;

private:
    float mRadius = 1.0f;
    bool mDiffuse = false;
    UInt mDiffuseTexure;
};
typedef SharedPtr<KiriSphere> KiriSpherePtr;
#endif