/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:36:30 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-20 03:25:11
 */
#include <kiri_core/model/model_sphere.h>

KiriSphere::KiriSphere()
{
    instancing = false;
    mesh = new KiriMeshSphere();
}

KiriSphere::KiriSphere(Array1<Matrix4x4F> _instMat4, bool _static_obj)
{
    instancing = true;
    static_obj = _static_obj;
    mesh = new KiriMeshSphere(_instMat4, static_obj);
}

KiriSphere::KiriSphere(float _radius)
{
    instancing = false;
    radius = _radius;
    mesh = new KiriMeshSphere(_radius);
}

KiriSphere::KiriSphere(float _radius, Int _xSeg, Int _ySeg, Array1<Matrix4x4F> _instMat4, bool _static_obj)
{
    instancing = true;
    static_obj = _static_obj;
    radius = _radius;
    mesh = new KiriMeshSphere(_radius, _xSeg, _ySeg, _instMat4, static_obj);
}

void KiriSphere::LoadDiffuseMap(UInt _diffuseTexure)
{
    diffuseTexure = _diffuseTexure;
}

void KiriSphere::SetDiffuseMap(bool _diffuse)
{
    diffuse = _diffuse;
}

void KiriSphere::Draw()
{
    if (diffuse)
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuseTexure);
    }

    KiriModel::Draw();
    mesh->Draw();
}