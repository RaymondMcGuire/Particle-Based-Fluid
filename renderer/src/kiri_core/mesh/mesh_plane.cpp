/*
 * @Author: Xu.Wang
 * @Date: 2020-03-17 17:24:27
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-25 13:11:32
 */
#include <kiri_core/mesh/mesh_plane.h>
#include <glad/glad.h>
void KiriMeshPlane::Construct()
{
    mDrawElem = false;
    mVertDataType = DataType::Standard;
    mVerticesNum = 6;
    for (size_t i = 0; i < mVerticesNum; i++)
    {
        AddVertStand(Vector3F(mPlaneVertices[i * 8 + 0], mPlaneVertices[i * 8 + 1], mPlaneVertices[i * 8 + 2]),
                     Vector3F(mPlaneVertices[i * 8 + 3], mPlaneVertices[i * 8 + 4], mPlaneVertices[i * 8 + 5]),
                     Vector2F(mPlaneVertices[i * 8 + 6], mPlaneVertices[i * 8 + 7]));
    }

    SetupVertex();
}

KiriMeshPlane::KiriMeshPlane()
{
    mInstance = false;

    Construct();
}

KiriMeshPlane::KiriMeshPlane(float _width, float _y, Vector3F _normal)
{
    mInstance = false;
    mWidth = _width;
    y = _y;
    mNormal = _normal;
    Array1<float> _planeVertices = {
        // positions            // normals         // texcoords
        mWidth, y, mWidth, mNormal.x, mNormal.y, mNormal.z, mWidth, 0.0f,
        -mWidth, y, mWidth, mNormal.x, mNormal.y, mNormal.z, 0.0f, 0.0f,
        -mWidth, y, -mWidth, mNormal.x, mNormal.y, mNormal.z, 0.0f, mWidth,

        mWidth, y, mWidth, mNormal.x, mNormal.y, mNormal.z, mWidth, 0.0f,
        -mWidth, y, -mWidth, mNormal.x, mNormal.y, mNormal.z, 0.0f, mWidth,
        mWidth, y, -mWidth, mNormal.x, mNormal.y, mNormal.z, mWidth, mWidth};
    mPlaneVertices = _planeVertices;

    Construct();
}

void KiriMeshPlane::Draw()
{
    glBindVertexArray(mVAO);
    glDrawArrays(GL_TRIANGLES, 0, (UInt)mVerticesNum);
    glBindVertexArray(0);
}