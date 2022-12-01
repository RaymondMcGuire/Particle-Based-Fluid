/*
 * @Author: Xu.Wang
 * @Date: 2020-03-25 15:29:03
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-25 15:31:17
 */

#include <kiri_core/model/model_cylinder.h>

KiriCylinder::KiriCylinder(float mBaseRadius, float mTopRadius, float height,
                           Int mSectorCount, Int mStackCount, bool mSmooth)
{
    mInstance = false;
    mMesh = new KiriMeshCylinder(mBaseRadius, mTopRadius, height,
                                 mSectorCount, mStackCount, mSmooth);
}

void KiriCylinder::Draw()
{

    KiriModel::Draw();
    mMesh->Draw();
}