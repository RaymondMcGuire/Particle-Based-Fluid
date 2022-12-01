/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-19 18:23:50
 * @LastEditTime: 2020-10-19 23:10:05
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\mesh\mesh_triangle.cpp
 * @Ref: https://github.com/manteapi/hokusai
 */

#include <kiri_core/mesh/mesh_triangle.h>

KiriMeshTriangle::KiriMeshTriangle()
{
    mVertices.clear();
    mNormals.clear();
    mTriangles.clear();
}

KiriMeshTriangle::KiriMeshTriangle(Array1Vec3F vertices, Array1Vec3F normals, Array1Vec3F triangles)
{
    mVertices = vertices;
    mNormals = normals;
    mTriangles = triangles;
}

KiriMeshTriangle::~KiriMeshTriangle()
{
}

void KiriMeshTriangle::GetEdges(Array1<std::pair<Int, Int>> &_edges) const
{
    std::set<std::pair<Int, Int>> tmpCleaningBag;
    std::pair<std::set<std::pair<Int, Int>>::iterator, bool> ret1, ret2;
    for (size_t i = 0; i < mTriangles.size(); ++i)
    {
        ret1 = tmpCleaningBag.insert(std::pair<Int, Int>((Int)mTriangles[i][0], (Int)mTriangles[i][1]));
        ret2 = tmpCleaningBag.insert(std::pair<Int, Int>((Int)mTriangles[i][1], (Int)mTriangles[i][0]));
        if (ret1.second == true && ret2.second == true)
            tmpCleaningBag.erase(ret2.first);
        if (ret1.second == false && ret2.second == true)
            tmpCleaningBag.erase(ret2.first);
        if (ret1.second == true && ret2.second == false)
            tmpCleaningBag.erase(ret1.first);

        ret1 = tmpCleaningBag.insert(std::pair<Int, Int>((Int)mTriangles[i][1], (Int)mTriangles[i][2]));
        ret2 = tmpCleaningBag.insert(std::pair<Int, Int>((Int)mTriangles[i][2], (Int)mTriangles[i][1]));
        if (ret1.second == true && ret2.second == true)
            tmpCleaningBag.erase(ret2.first);
        if (ret1.second == false && ret2.second == true)
            tmpCleaningBag.erase(ret2.first);
        if (ret1.second == true && ret2.second == false)
            tmpCleaningBag.erase(ret1.first);

        ret1 = tmpCleaningBag.insert(std::pair<Int, Int>((Int)mTriangles[i][2], (Int)mTriangles[i][0]));
        ret2 = tmpCleaningBag.insert(std::pair<Int, Int>((Int)mTriangles[i][0], (Int)mTriangles[i][2]));
        if (ret1.second == true && ret2.second == true)
            tmpCleaningBag.erase(ret2.first);
        if (ret1.second == false && ret2.second == true)
            tmpCleaningBag.erase(ret2.first);
        if (ret1.second == true && ret2.second == false)
            tmpCleaningBag.erase(ret1.first);
    }
    for (std::set<std::pair<Int, Int>>::iterator it = tmpCleaningBag.begin(); it != tmpCleaningBag.end(); ++it)
        _edges.append(*it);
}