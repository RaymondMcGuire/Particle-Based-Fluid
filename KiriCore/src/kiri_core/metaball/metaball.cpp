/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-30 19:52:50
 * @LastEditTime: 2020-12-30 20:19:15
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\metaball\metaball.cpp
 */

#include <kiri_core/metaball/metaball.h>

void KiriMetaBall::PushMetaBall(Vector3F Center, float Radius)
{
    mMetaBallArray.append(MetaBallData(Center, Radius));
}

float KiriMetaBall::MetaBallStandardFunc(Vector3F Position, Vector3F Center, float Radius)
{
    Vector3F distance = Position - Center;
    return Radius * Radius / (distance.lengthSquared() + MEpsilon<float>());
}

float KiriMetaBall::Sampling(Vector3F Position)
{
    float r = 0.f;

    for (size_t i = 0; i < mMetaBallArray.size(); i++)
    {
        r += MetaBallStandardFunc(Position, mMetaBallArray[i].center, mMetaBallArray[i].radius);
    }

    return r - mEquipotentialValue;
}
