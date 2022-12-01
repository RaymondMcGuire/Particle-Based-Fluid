/*** 
 * @Author: Xu.WANG
 * @Date: 2021-04-07 14:05:06
 * @LastEditTime: 2021-04-08 15:54:53
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\mesh\mesh_line.cpp
 */
#include <kiri_core/mesh/mesh_line.h>

void KiriMeshLine::ConvertEdge2Vert(const Array1<KiriEdge> &edges)
{
    for (size_t i = 0; i < edges.size(); i++)
    {
        AddVertSimple(edges[i].start, edges[i].col);
        AddVertSimple(edges[i].end, edges[i].col);
    }
}

void KiriMeshLine::Construct()
{
    SetupVertex();
}

void KiriMeshLine::Draw()
{
    glBindVertexArray(mVAO);
    glLineWidth(3.f);
    glDrawArrays(GL_LINES, 0, (UInt)mVerticesNum);
    glBindVertexArray(0);
}