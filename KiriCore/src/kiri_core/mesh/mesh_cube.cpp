/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:32:41 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-25 17:59:40
 */
#include <kiri_core/mesh/mesh_cube.h>

void KiriMeshCube::Construct()
{
    drawElem = false;
    vertDataType = DataType::Standard;
    verticesNum = 36;
    for (size_t i = 0; i < verticesNum; i++)
    {

        addVertStand(Vector3F(cubeVertices[i * 8 + 0], cubeVertices[i * 8 + 1], cubeVertices[i * 8 + 2]),
                     Vector3F(cubeVertices[i * 8 + 3], cubeVertices[i * 8 + 4], cubeVertices[i * 8 + 5]),
                     Vector2F(cubeVertices[i * 8 + 6], cubeVertices[i * 8 + 7]));
    }

    SetupVertex();
}

KiriMeshCube::KiriMeshCube()
{
    instancing = false;

    Construct();
}

void KiriMeshCube::Draw()
{
    glBindVertexArray(mVAO);
    glDrawArrays(GL_TRIANGLES, 0, (UInt)verticesNum);
    glBindVertexArray(0);
}