/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:37:34 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-25 18:00:23
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

    float getSide()
    {
        return side;
    }

    void Draw() override;

private:
    float side = 1.0f;
    bool imgMode;
    void Construct() override;

    Array1<float> quadPos = {
        -side, side, 0.0f,
        -side, -side, 0.0f,
        side, side, 0.0f,
        side, -side, 0.0f};

    Array1<float> quadTexCoord = {
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f,
        1.0f,
        0.0f};

    Array1<float> quadColor = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,

        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 1.0f};
};
#endif
