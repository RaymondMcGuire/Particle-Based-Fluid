/*
 * @Author: Xu.Wang
 * @Date: 2020-03-17 17:36:10
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-15 01:45:52
 */
#include <kiri_core/model/model_plane.h>

KiriPlane::KiriPlane()
{
    mMesh = new KiriMeshPlane();
    mWidth = ((KiriMeshPlane *)mMesh)->GetWidth();
    y = ((KiriMeshPlane *)mMesh)->GetY();
    mNormal = ((KiriMeshPlane *)mMesh)->GetNormal();
}

KiriPlane::KiriPlane(float _width, float _y, Vector3F _normal)
{
    mWidth = _width;
    y = _y;
    mNormal = _normal;
    mMesh = new KiriMeshPlane(mWidth, y, mNormal);
}

void KiriPlane::Draw()
{
    KiriModel::Draw();
    mMesh->Draw();
}