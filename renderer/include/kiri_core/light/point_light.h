/***
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 19:46:15
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \KiriCore\include\kiri_core\light\point_light.h
 */

#ifndef _KIRI_POINT_LIGHT_H_
#define _KIRI_POINT_LIGHT_H_
#pragma once
#include <kiri_core/light/light.h>
#include <kiri_core/material/material_lamp.h>
#include <kiri_core/model/model_cube.h>

class KiriPointLight : public KiriLight
{
public:
    KiriPointLight();
    KiriPointLight(Vector3F, Vector3F);

    Vector3F position;

    Vector3F ambient = Vector3F(0.05f, 0.05f, 0.05f);
    Vector3F diffuse;
    Vector3F specular = Vector3F(1.0f, 1.0f, 1.0f);

    float constant = 1.0f;
    float linear = 0.09f;
    float quadratic = 0.032f;

    void Draw() override;

private:
    KiriMaterialLampPtr material;
    void SetModel();
};

typedef SharedPtr<KiriPointLight> KiriPointLightPtr;
#endif