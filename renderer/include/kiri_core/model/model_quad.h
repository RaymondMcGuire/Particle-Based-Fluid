/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:35:55
 * @FilePath: \core\include\kiri_core\model\model_quad.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _KIRI_MODEL_QUAD_H_
#define _KIRI_MODEL_QUAD_H_
#pragma once
#include <kiri_core/model/model_internal.h>
#include <kiri_core/mesh/mesh_quad.h>

class KiriQuad : public KiriModelInternal
{
public:
    KiriQuad();
    KiriQuad(float);
    KiriQuad(Array1<Vector2F>);
    KiriQuad(float, Array1<Vector2F>);

    void Draw() override;

private:
    bool mImgMode = true;

    float mSide = 1.0f;
};
typedef SharedPtr<KiriQuad> KiriQuadPtr;
#endif