/*
 * @Author: Xu.Wang
 * @Date: 2020-03-17 17:36:30
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-20 03:25:11
 */
#include <kiri_core/model/model_sphere.h>

KiriSphere::KiriSphere()
{
    mInstance = false;
    mMesh = new KiriMeshSphere();
}

KiriSphere::KiriSphere(Array1<Matrix4x4F> _instMat4, bool _static_obj)
{
    mInstance = true;
    mStaticObj = _static_obj;
    mMesh = new KiriMeshSphere(_instMat4, mStaticObj);
}

KiriSphere::KiriSphere(float _radius)
{
    mInstance = false;
    mRadius = _radius;
    mMesh = new KiriMeshSphere(_radius);
}

KiriSphere::KiriSphere(float _radius, Int _xSeg, Int _ySeg, Array1<Matrix4x4F> _instMat4, bool _static_obj)
{
    mInstance = true;
    mStaticObj = _static_obj;
    mRadius = _radius;
    mMesh = new KiriMeshSphere(_radius, _xSeg, _ySeg, _instMat4, mStaticObj);
}

void KiriSphere::LoadDiffuseMap(UInt _diffuseTexure)
{
    mDiffuseTexure = _diffuseTexure;
}

void KiriSphere::SetDiffuseMap(bool _diffuse)
{
    mDiffuse = _diffuse;
}

void KiriSphere::Draw()
{
    if (mDiffuse)
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, mDiffuseTexure);
    }

    KiriModel::Draw();
    mMesh->Draw();
}