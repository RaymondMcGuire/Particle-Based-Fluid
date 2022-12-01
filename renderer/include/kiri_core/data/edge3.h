/*** 
 * @Author: Xu.WANG
 * @Date: 2021-03-27 01:49:01
 * @LastEditTime: 2021-06-07 18:02:36
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\data\edge3.h
 */

#ifndef _KIRI_EDGE3_H_
#define _KIRI_EDGE3_H_

#pragma once

#include <kiri_core/data/vertex3.h>

namespace KIRI
{
    class KiriEdge3
    {
    public:
        explicit KiriEdge3(UInt id, const KiriVertex3Ptr &ori, const KiriVertex3Ptr &dst)
        {
            // mNext = NULL;
            // mPrev = NULL;
            // mTwin = NULL;

            mId = id;
            mOrigin = ori;
            mDest = dst;

            mNextId = mPrevId = mTwinId = -1;
        }

        ~KiriEdge3() noexcept {}

        constexpr UInt GetId() const { return mId; }
        void SetId(UInt id) { mId = id; }

        // void SetNextEdge(SharedPtr<KiriEdge3> nxt) { mNext = nxt; }
        // void SetPrevEdge(SharedPtr<KiriEdge3> pre) { mPrev = pre; }
        // void SetTwinEdge(SharedPtr<KiriEdge3> twn) { mTwin = twn; }

        // const SharedPtr<KiriEdge3> &GetNextEdge() const { return mNext; }
        // const SharedPtr<KiriEdge3> &GetPrevEdge() const { return mPrev; }
        // const SharedPtr<KiriEdge3> &GetTwinEdge() const { return mTwin; }

        void SetNextEdgeId(UInt nxt) { mNextId = nxt; }
        void SetPrevEdgeId(UInt pre) { mPrevId = pre; }
        void SetTwinEdgeId(UInt twn) { mTwinId = twn; }

        UInt GetNextEdgeId() const { return mNextId; }
        UInt GetPrevEdgeId() const { return mPrevId; }
        UInt GetTwinEdgeId() const { return mTwinId; }

        const KiriVertex3Ptr &GetOriginVertex() const { return mOrigin; }
        const KiriVertex3Ptr &GetDestVertex() const { return mDest; }

        const bool IsEqual(const KiriVertex3Ptr &a, const KiriVertex3Ptr &b) const
        {
            return ((mOrigin->IsEqual(a) && mDest->IsEqual(b)) || (mOrigin->IsEqual(b) && mDest->IsEqual(a)));
        }

        const bool IsEqual(const SharedPtr<KiriEdge3> &edge) const
        {
            return ((mOrigin->IsEqual(edge->GetOriginVertex()) && mDest->IsEqual(edge->GetDestVertex())) || (mOrigin->IsEqual(edge->GetDestVertex()) && mDest->IsEqual(edge->GetOriginVertex())));
        }

        void PrintEdgeInfo()
        {
            KIRI_LOG_DEBUG("----------EDGE INFO----------");
            KIRI_LOG_DEBUG("edge info: id={0}, origin=({1},{2},{3}), dest=({4},{5},{6})",
                           mId,
                           mOrigin->GetValue().x, mOrigin->GetValue().y, mOrigin->GetValue().z,
                           mDest->GetValue().x, mDest->GetValue().y, mDest->GetValue().z);

            // KIRI_LOG_DEBUG("prev edge id={0}, next edge id={1}, twin edge id={2},",
            //                (mPrev != NULL) ? mPrev->GetId() : -1,
            //                (mNext != NULL) ? mNext->GetId() : -1,
            //                (mTwin != NULL) ? mTwin->GetId() : -1);

            KIRI_LOG_DEBUG("edge id={0}, prev edge id={1}, next edge id={2}, twin edge id={3};",
                           mId,
                           mPrevId,
                           mNextId,
                           mTwinId);
            KIRI_LOG_DEBUG("------------------------------");
        }

        void CleanEdge()
        {
            // mPrev = NULL;
            // mNext = NULL;
            // mTwin = NULL;
            mId = -1;
            mNextId = mPrevId = mTwinId = -1;
        }

    private:
        UInt mId, mNextId, mPrevId, mTwinId;
        KiriVertex3Ptr mOrigin, mDest;
        // SharedPtr<KiriEdge3> mNext;
        // SharedPtr<KiriEdge3> mPrev;
        // SharedPtr<KiriEdge3> mTwin;
    };
    typedef SharedPtr<KiriEdge3> KiriEdge3Ptr;
} // namespace KIRI

#endif /* _KIRI_EDGE3_H_ */