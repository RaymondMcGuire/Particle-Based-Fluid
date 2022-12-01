/***
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:35:39
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \KiriCore\include\kiri_core\model\model_cylinder.h
 */

#ifndef _KIRI_MODEL_CYLINDER_H_
#define _KIRI_MODEL_CYLINDER_H_
#pragma once
#include <kiri_core/model/model_internal.h>
#include <kiri_core/mesh/mesh_cylinder.h>

class KiriCylinder : public KiriModelInternal
{
public:
    KiriCylinder(float mBaseRadius = 1.0f, float mTopRadius = 1.0f, float height = 2.0f,
                 Int mSectorCount = 36, Int mStackCount = 8, bool mSmooth = true);

    void Draw() override;
};
typedef SharedPtr<KiriCylinder> KiriCylinderPtr;
#endif