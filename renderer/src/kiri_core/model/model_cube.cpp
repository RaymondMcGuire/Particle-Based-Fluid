/*
 * @Author: Xu.Wang
 * @Date: 2020-03-17 17:34:54
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-20 04:22:48
 */
#include <kiri_core/model/model_cube.h>

KiriCube::KiriCube()
{
    mMesh = new KiriMeshCube();
}

void KiriCube::LoadDiffuseMap(UInt _diffuseTexure)
{
    diffuseTexure = _diffuseTexure;
}

void KiriCube::SetDiffuseMap(bool _diffuse)
{
    mDiffuse = _diffuse;
}

void KiriCube::SetRenderOutside(bool _outside)
{
    outside = _outside;
}

void KiriCube::Draw()
{
    KiriModel::Draw();

    if (!outside)
        glDisable(GL_CULL_FACE);

    if (mDiffuse)
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuseTexure);
    }

    if (bWireFrame)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
    mMesh->Draw();
    if (bWireFrame)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    if (!outside)
        glEnable(GL_CULL_FACE);
}