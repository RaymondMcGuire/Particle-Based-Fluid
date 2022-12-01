/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2021-03-29 19:04:30
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 16:21:38
 * @FilePath: \Kiri\KiriCore\src\kiri_core\model\model_box.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_core/model/model_box.h>

void KiriBox::ConstructBox(const Vector3F &center, const Vector3F &side)
{
    mCenter = center;
    mSide = side;

    ResetModelMatrix();
    Translate(mCenter);
    Scale(mSide / 2.f);
}

void KiriBox::ResetBox(const Vector3F &center, const Vector3F & side)
{
    mCenter = center;
    mSide = side;

    ResetModelMatrix();
    Translate(mCenter);
    Scale(mSide / 2.f);
}
