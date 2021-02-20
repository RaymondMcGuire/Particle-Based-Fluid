/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-30 19:35:37
 * @LastEditTime: 2021-02-20 19:46:32
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\metaball\metaball.h
 */

#ifndef _KIRI_META_BALL_H_
#define _KIRI_META_BALL_H_
#pragma once
#include <kiri_pch.h>

struct MetaBallData
{
    Vector3F center;
    float radius;

    MetaBallData(const Vector3F &Center, const float Radius) : center(Center), radius(Radius) {}
};

class KiriMetaBall
{
public:
    KiriMetaBall() : mEquipotentialValue(1.f){};
    KiriMetaBall(float EquipotentialValue) : mEquipotentialValue(EquipotentialValue){};

    void PushMetaBall(Vector3F Center, float Radius);
    float Sampling(Vector3F Position);

private:
    float MetaBallStandardFunc(Vector3F Position, Vector3F Center, float Radius);

    float mEquipotentialValue;
    Array1<MetaBallData> mMetaBallArray;
};

typedef SharedPtr<KiriMetaBall> KiriMetaBallPtr;

#endif
