/***
 * @Author: Xu.WANG
 * @Date: 2021-02-22 11:23:43
 * @LastEditTime: 2021-04-07 14:09:53
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriCore\src\kiri_core\mesh\mesh_cube.cpp
 */

#include <kiri_core/mesh/mesh_cube.h>

void KiriMeshCube::Construct()
{
    mDrawElem = false;
    mVertDataType = DataType::Standard;
    mVerticesNum = 36;
    for (size_t i = 0; i < mVerticesNum; i++)
    {

        AddVertStand(Vector3F(mCubeVertices[i * 8 + 0], mCubeVertices[i * 8 + 1], mCubeVertices[i * 8 + 2]),
                     Vector3F(mCubeVertices[i * 8 + 3], mCubeVertices[i * 8 + 4], mCubeVertices[i * 8 + 5]),
                     Vector2F(mCubeVertices[i * 8 + 6], mCubeVertices[i * 8 + 7]));
    }

    SetupVertex();
}

KiriMeshCube::KiriMeshCube()
{
    mInstance = false;

    Construct();
}

void KiriMeshCube::Draw()
{
    glBindVertexArray(mVAO);
    glDrawArrays(GL_TRIANGLES, 0, (UInt)mVerticesNum);
    glBindVertexArray(0);
}