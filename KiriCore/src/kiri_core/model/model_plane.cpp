/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:36:10 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-15 01:45:52
 */
#include <kiri_core/model/model_plane.h>

KiriPlane::KiriPlane()
{
    mesh = new KiriMeshPlane();
    width = ((KiriMeshPlane *)mesh)->getWidth();
    y = ((KiriMeshPlane *)mesh)->getY();
    normal = ((KiriMeshPlane *)mesh)->getNormal();
}

KiriPlane::KiriPlane(float _width, float _y, Vector3F _normal)
{
    width = _width;
    y = _y;
    normal = _normal;
    mesh = new KiriMeshPlane(width, y, normal);
}

void KiriPlane::Draw()
{
    KiriModel::Draw();
    mesh->Draw();
}