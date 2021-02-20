/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:36:19 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:36:19 
 */
#include <kiri_core/model/model_quad.h>

KiriQuad::KiriQuad()
{
    mesh = new KiriMeshQuad();
    side = ((KiriMeshQuad *)mesh)->getSide();
}

KiriQuad::KiriQuad(float _side)
{
    side = _side;
    mesh = new KiriMeshQuad(side);
}

KiriQuad::KiriQuad(Array1<Vector2F> _instVec2)
{
    mesh = new KiriMeshQuad(_instVec2);
}

KiriQuad::KiriQuad(float _side, Array1<Vector2F> _instVec2)
{
    side = _side;
    mesh = new KiriMeshQuad(side, _instVec2);
}

void KiriQuad::Draw()
{
    KiriModel::Draw();
    mesh->Draw();
}