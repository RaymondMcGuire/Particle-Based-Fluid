/***
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-04-07 15:24:03
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriCore\include\kiri_core\model\model_line.h
 */

#ifndef _KIRI_MODEL_LINE_H_
#define _KIRI_MODEL_LINE_H_
#pragma once
#include <kiri_core/model/model_internal.h>
#include <kiri_core/mesh/mesh_line.h>
class KiriLine : public KiriModelInternal
{
public:
    explicit KiriLine(
        const Array1<KiriEdge> &edges)
    {
        mMesh = new KiriMeshLine(edges);
    }

    KiriLine(const KiriLine &) = delete;
    KiriLine &operator=(const KiriLine &) = delete;
    ~KiriLine() noexcept {}

    void Draw() override;
};
typedef SharedPtr<KiriLine> KiriLinePtr;
#endif