/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 19:45:53
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\light\direction_light.h
 */

#ifndef _KIRI_DIRECTION_LIGHT_H_
#define _KIRI_DIRECTION_LIGHT_H_
#pragma once
#include <kiri_core/light/light.h>

class KiriDirectionLight : public KiriLight
{
public:
    KiriDirectionLight(Vector3F, Vector3F);
    ~KiriDirectionLight();

    Vector3F direction;

    Vector3F ambient = Vector3F(0.05f, 0.05f, 0.05f);
    Vector3F diffuse;
    Vector3F specular = Vector3F(1.0f, 1.0f, 1.0f);
};
#endif