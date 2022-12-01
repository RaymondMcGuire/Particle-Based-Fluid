/*** 
 * @Author: Xu.WANG
 * @Date: 2021-02-22 18:33:21
 * @LastEditTime: 2021-06-07 14:33:52
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri2D\Kiri2d\src\kiri_core\data\face3.cpp
 */

#include <kiri_core/data/face3.h>

namespace KIRI
{
    void KiriFace3::OrientFace(const KiriVertex3Ptr &orient)
    {
        if (!CheckFaceDir(orient->GetValue()))
            ReverseFaceDir();
    }

    void KiriFace3::PrintFaceInfo()
    {
        KIRI_LOG_DEBUG("----------FACET INFO----------");
        KIRI_LOG_DEBUG("face idx={0}", mIdx);
        KIRI_LOG_DEBUG("face info: v1=({0},{1},{2}),v2=({3},{4},{5}),v3=({6},{7},{8}),mNormal=({9},{10},{11})",
                       mVertices[0]->GetValue().x, mVertices[0]->GetValue().y, mVertices[0]->GetValue().z,
                       mVertices[1]->GetValue().x, mVertices[1]->GetValue().y, mVertices[1]->GetValue().z,
                       mVertices[2]->GetValue().x, mVertices[2]->GetValue().y, mVertices[2]->GetValue().z,
                       mNormal.x, mNormal.y, mNormal.z);

        // for (size_t i = 0; i < 3; i++)
        //     mEdges[i]->PrintEdgeInfo();

        KIRI_LOG_DEBUG("------------------------------");
    }

    void KiriFace3::ReverseFaceDir()
    {
        auto tmp = mVertices[1];
        mVertices[1] = mVertices[2];
        mVertices[2] = tmp;
        mNormal *= -1.f;
    }

    void KiriFace3::GenerateEdges()
    {
        // mEdges[0] = std::make_shared<KiriEdge3>(mIdx * 3, mVertices[0], mVertices[1]);
        // mEdges[1] = std::make_shared<KiriEdge3>(mIdx * 3 + 1, mVertices[1], mVertices[2]);
        // mEdges[2] = std::make_shared<KiriEdge3>(mIdx * 3 + 2, mVertices[2], mVertices[0]);

        // mEdges[0]->SetNextEdge(mEdges[1]);
        // mEdges[1]->SetNextEdge(mEdges[2]);
        // mEdges[2]->SetNextEdge(mEdges[0]);

        // mEdges[0]->SetPrevEdge(mEdges[2]);
        // mEdges[1]->SetPrevEdge(mEdges[0]);
        // mEdges[2]->SetPrevEdge(mEdges[1]);

        mEdges[0] = mIdx * 3;
        mEdges[1] = mIdx * 3 + 1;
        mEdges[2] = mIdx * 3 + 2;
    }

    // const KiriEdge3Ptr &KiriFace3::GetEdge(const KiriVertex3Ptr &a, const KiriVertex3Ptr &b)
    // {
    //     for (size_t i = 0; i < 3; i++)
    //     {
    //         if (mEdges[i]->IsEqual(a, b))
    //             return mEdges[i];
    //     }
    //     return NULL;
    // }

    // void KiriFace3::LinkEdge(const KiriEdge3Ptr &e)
    // {
    //     auto cur_edge = this->GetEdge(e->GetOriginVertex(), e->GetDestVertex());
    //     if (cur_edge == NULL)
    //     {
    //         KIRI_LOG_ERROR("LinkEdge ERROR: Current edge is not exist, cannot connect edges!");
    //         return;
    //     }

    //     if (cur_edge->GetId() == e->GetId())
    //         KIRI_LOG_ERROR("LinkEdge ERROR:edge idx={0}, face idx={1}", cur_edge->GetId(), mIdx);

    //     e->SetTwinEdge(cur_edge);
    //     cur_edge->SetTwinEdge(e);
    // }

    // void KiriFace3::LinkFace(const KiriFace3Ptr &f, const KiriVertex3Ptr &a, const KiriVertex3Ptr &b)
    // {
    //     auto twin = f->GetEdge(a, b);
    //     if (twin == NULL)
    //     {
    //         KIRI_LOG_ERROR("LinkFace ERROR: Twin edge is not exist, cannot connect edges!");
    //         return;
    //     }

    //     auto cur_edge = this->GetEdge(a, b);
    //     if (cur_edge != NULL)
    //     {
    //         twin->SetTwinEdge(cur_edge);
    //         cur_edge->SetTwinEdge(twin);
    //     }
    // }

    // void KiriFace3::LinkFace(const KiriFace3Ptr &f)
    // {
    //     for (size_t i = 0; i < 3; i++)
    //     {
    //         auto twin = f->GetEdge(mEdges[i]->GetOriginVertex(), mEdges[i]->GetDestVertex());
    //         if (twin != NULL)
    //         {
    //             mEdges[i]->SetTwinEdge(twin);
    //             twin->SetTwinEdge(mEdges[i]);
    //         }
    //     }
    // }
}