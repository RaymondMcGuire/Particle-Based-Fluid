/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:27:30
 * @FilePath: \core\include\kiri_core\model\model_box.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2021-03-29 19:04:30
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 16:21:17
 * @FilePath: \Kiri\KiriCore\include\kiri_core\model\model_box.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _KIRI_MODEL_BOX_H_
#define _KIRI_MODEL_BOX_H_
#pragma once
#include <kiri_core/model/model_cube.h>

class KiriBox : public KiriCube
{
public:
    explicit KiriBox(
        const Vector3F &center = Vector3F(0.f),
        const Vector3F &mSide = Vector3F(1.f))
        : mCenter(center), mSide(mSide)
    {
        ConstructBox(mCenter, mSide);
    }

    ~KiriBox() noexcept {}

    void ResetBox(const Vector3F &center, const Vector3F &mSide);

protected:
    void ConstructBox(const Vector3F &center, const Vector3F &mSide);

private:
    Vector3F mCenter;
    Vector3F mSide;
};
typedef SharedPtr<KiriBox> KiriBoxPtr;
#endif