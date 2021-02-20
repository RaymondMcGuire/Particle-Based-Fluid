/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 19:37:12
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\model\model_quad.h
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
    bool img_mode = true;

    float side = 1.0f;
};
typedef SharedPtr<KiriQuad> KiriQuadPtr;
#endif