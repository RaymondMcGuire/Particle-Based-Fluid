/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:36:36
 * @FilePath: \core\include\kiri_core\model\model_plane.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _KIRI_MODEL_PLANE_H_
#define _KIRI_MODEL_PLANE_H_
#pragma once
#include <kiri_core/model/model_internal.h>
#include <kiri_core/mesh/mesh_plane.h>

class KiriPlane : public KiriModelInternal
{
public:
    KiriPlane();
    KiriPlane(float, float, Vector3F);

    void Draw() override;

private:
    float mWidth;
    float y;
    Vector3F mNormal;
};

typedef SharedPtr<KiriPlane> KiriPlanePtr;

#endif