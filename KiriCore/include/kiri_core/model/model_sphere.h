/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:37:29
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\model\model_sphere.h
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
    float radius = 1.0f;
    bool diffuse = false;
    UInt diffuseTexure;
};
typedef SharedPtr<KiriSphere> KiriSpherePtr;
#endif