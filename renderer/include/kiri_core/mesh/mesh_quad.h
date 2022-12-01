/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:46:16
 * @FilePath: \core\include\kiri_core\mesh\mesh_quad.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _KIRI_MESH_QUAD_H_
#define _KIRI_MESH_QUAD_H_

#include <kiri_core/mesh/mesh_internal.h>

class KiriMeshQuad : public KiriMeshInternal
{
public:
    KiriMeshQuad();
    KiriMeshQuad(float);
    KiriMeshQuad(float, Array1<Vector2F>);
    KiriMeshQuad(Array1<Vector2F>);
    ~KiriMeshQuad(){};

    float GetSide()
    {
        return mSide;
    }

    void Draw() override;

private:
    float mSide = 1.0f;
    bool mImgMode;
    void Construct() override;

    Array1<float> mQuadPos = {
        -mSide, mSide, 0.0f,
        -mSide, -mSide, 0.0f,
        mSide, mSide, 0.0f,
        mSide, -mSide, 0.0f};

    Array1<float> mQuadTexCoord = {
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f,
        1.0f,
        0.0f};

    Array1<float> mQuadColor = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,

        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 1.0f};
};
#endif
