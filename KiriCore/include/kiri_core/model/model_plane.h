/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 19:37:07
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\model\model_plane.h
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
    float width;
    float y;
    Vector3F normal;
};

typedef SharedPtr<KiriPlane> KiriPlanePtr;

#endif