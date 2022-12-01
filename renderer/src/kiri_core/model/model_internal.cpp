/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:34:58 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-20 04:02:06
 */
#include <kiri_core/model/model_internal.h>

void KiriModelInternal::UpdateInstance(Array1<Matrix4x4F> _instMat4)
{
    mMesh->UpdateInstance(_instMat4);
}

void KiriModelInternal::SetWireFrame(bool wf)
{
    bWireFrame = wf;
}