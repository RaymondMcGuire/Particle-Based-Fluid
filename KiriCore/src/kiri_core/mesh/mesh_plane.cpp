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
    drawElem = false;
    vertDataType = DataType::Standard;
    verticesNum = 6;
    for (size_t i = 0; i < verticesNum; i++)
    {
        addVertStand(Vector3F(planeVertices[i * 8 + 0], planeVertices[i * 8 + 1], planeVertices[i * 8 + 2]),
                     Vector3F(planeVertices[i * 8 + 3], planeVertices[i * 8 + 4], planeVertices[i * 8 + 5]),
                     Vector2F(planeVertices[i * 8 + 6], planeVertices[i * 8 + 7]));
    }

    SetupVertex();
}

KiriMeshPlane::KiriMeshPlane()
{
    instancing = false;

    Construct();
}

KiriMeshPlane::KiriMeshPlane(float _width, float _y, Vector3F _normal)
{
    instancing = false;
    width = _width;
    y = _y;
    normal = _normal;
    Array1<float> _planeVertices = {
        // positions            // normals         // texcoords
        width, y, width, normal.x, normal.y, normal.z, width, 0.0f,
        -width, y, width, normal.x, normal.y, normal.z, 0.0f, 0.0f,
        -width, y, -width, normal.x, normal.y, normal.z, 0.0f, width,

        width, y, width, normal.x, normal.y, normal.z, width, 0.0f,
        -width, y, -width, normal.x, normal.y, normal.z, 0.0f, width,
        width, y, -width, normal.x, normal.y, normal.z, width, width};
    planeVertices = _planeVertices;

    Construct();
}

void KiriMeshPlane::Draw()
{
    glBindVertexArray(mVAO);
    glDrawArrays(GL_TRIANGLES, 0, (UInt)verticesNum);
    glBindVertexArray(0);
}