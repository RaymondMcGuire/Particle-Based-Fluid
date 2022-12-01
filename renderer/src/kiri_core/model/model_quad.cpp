/*
 * @Author: Xu.Wang
 * @Date: 2020-03-17 17:36:19
 * @Last Modified by:   Xu.Wang
 * @Last Modified time: 2020-03-17 17:36:19
 */
#include <kiri_core/model/model_quad.h>

KiriQuad::KiriQuad()
{
    mMesh = new KiriMeshQuad();
    mSide = ((KiriMeshQuad *)mMesh)->GetSide();
}

KiriQuad::KiriQuad(float _side)
{
    mSide = _side;
    mMesh = new KiriMeshQuad(mSide);
}

KiriQuad::KiriQuad(Array1<Vector2F> _instVec2)
{
    mMesh = new KiriMeshQuad(_instVec2);
}

KiriQuad::KiriQuad(float _side, Array1<Vector2F> _instVec2)
{
    mSide = _side;
    mMesh = new KiriMeshQuad(mSide, _instVec2);
}

void KiriQuad::Draw()
{
    KiriModel::Draw();
    mMesh->Draw();
}