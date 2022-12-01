/***
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 19:46:10
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \KiriCore\include\kiri_core\light\spot_light.h
 */

#ifndef _KIRI_SPOT_LIGHT_H
#define _KIRI_SPOT_LIGHT_H
#pragma once
#include <kiri_core/light/light.h>

class KiriSpotLight : public KiriLight
{
public:
    KiriSpotLight();
    KiriSpotLight(Vector3F, Vector3F, Vector3F);
    ~KiriSpotLight();

    Vector3F position;
    Vector3F direction;

    Vector3F ambient = Vector3F(0.05f, 0.05f, 0.05f);
    Vector3F mDiffuse;
    Vector3F specular = Vector3F(1.0f, 1.0f, 1.0f);

    float constant = 1.0f;
    float linear = 0.09f;
    float quadratic = 0.032f;
};
#endif