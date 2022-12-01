/***
 * @Author: Xu.WANG
 * @Date: 2021-12-28 12:10:44
 * @LastEditTime: 2022-01-18 18:05:17
 * @LastEditors: Xu.WANG
 * @Description:
 */
#include <kiri_core/mesh/mesh_custom.h>

void KiriMeshCustom::Construct()
{
    mDrawElem = true;
    mVertDataType = DataType::Standard;
}

KiriMeshCustom::KiriMeshCustom()
{
    mInstance = false;

    Construct();
}

void KiriMeshCustom::Generate(std::vector<Vector3D> pos, std::vector<Vector3D> mNormal, std::vector<int> indices)
{
    mVerticesNum = pos.size();
    for (auto i = 0; i < mVerticesNum; i++)
    {

        AddVertStand(Vector3F(pos[i].x, pos[i].y, pos[i].z),
                     Vector3F(mNormal[i].x, mNormal[i].y, mNormal[i].z),
                     Vector2F(0.f, 1.f));
    }

    for (auto i = 0; i < indices.size(); i++)
        mIndices.append(indices[i]);

    SetupVertex();
}

void KiriMeshCustom::Draw()
{
    glBindVertexArray(mVAO);
    glDrawArrays(GL_TRIANGLES, 0, (UInt)mVerticesNum);
    glBindVertexArray(0);
}