/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-25 15:29:03 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-25 15:31:17
 */

#include <kiri_core/model/model_cylinder.h>

KiriCylinder::KiriCylinder(float baseRadius, float topRadius, float height,
                           Int sectorCount, Int stackCount, bool smooth)
{
    instancing = false;
    mesh = new KiriMeshCylinder(baseRadius, topRadius, height,
                                sectorCount, stackCount, smooth);
}

void KiriCylinder::Draw()
{

    KiriModel::Draw();
    mesh->Draw();
}