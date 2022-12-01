/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-04-08 15:54:38
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\mesh\mesh_line.h
 */

#ifndef _KIRI_MESH_LINE_H_
#define _KIRI_MESH_LINE_H_
#pragma once
#include <kiri_core/mesh/mesh_internal.h>

struct KiriEdge
{
    Vector3F start;
    Vector3F end;
    Vector3F col;

    KiriEdge(Vector3F start, Vector3F end, Vector3F col) : start(start), end(end), col(col) {}
};

class KiriMeshLine : public KiriMeshInternal
{
public:
    explicit KiriMeshLine(
        const Array1<KiriEdge> &edges)
    {
        mInstance = false;
        mDrawElem = false;
        mVertDataType = DataType::Simple;
        mVerticesNum = 0;

        ConvertEdge2Vert(edges);
        Construct();
    }

    KiriMeshLine(const KiriMeshLine &) = delete;
    KiriMeshLine &operator=(const KiriMeshLine &) = delete;
    ~KiriMeshLine() noexcept {}

    void Draw() override;
    virtual void Construct() override;

private:
    void ConvertEdge2Vert(const Array1<KiriEdge> &edges);
};
#endif
