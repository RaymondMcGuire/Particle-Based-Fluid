/*** 
 * @Author: Xu.WANG
 * @Date: 2021-03-27 01:49:01
 * @LastEditTime: 2021-06-07 14:44:23
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri2D\Kiri2d\include\kiri_core\data\face3.h
 */

#ifndef _KIRI_FACE3_H_
#define _KIRI_FACE3_H_

#pragma once

#include <kiri_core/data/edge3.h>

namespace KIRI
{
    class KiriFace3
    {
    public:
        explicit KiriFace3() : KiriFace3(std::make_shared<KiriVertex3>(), std::make_shared<KiriVertex3>(), std::make_shared<KiriVertex3>()) {}

        explicit KiriFace3(const KiriVertex3Ptr &a, const KiriVertex3Ptr &b, const KiriVertex3Ptr &c)
        {
            mIdx = -1;
            mVisible = false;

            mVertices[0] = a;
            mVertices[1] = b;
            mVertices[2] = c;

            mNormal = (-((b->GetValue() - a->GetValue()).cross(c->GetValue() - b->GetValue()))).normalized();
        }

        explicit KiriFace3(const KiriVertex3Ptr &a, const KiriVertex3Ptr &b, const KiriVertex3Ptr &c, const KiriVertex3Ptr &orient)
            : KiriFace3(a, b, c)
        {
            OrientFace(orient);
        }

        ~KiriFace3() noexcept {}

        inline constexpr UInt GetIdx() { return mIdx; }
        inline constexpr bool GetVisible() { return mVisible; }
        inline constexpr Int GetEdgeCount() const { return 3; }

        inline const Vector3F &GetNormal() const { return mNormal; }
        //inline const KiriEdge3Ptr &GetEdgesByIdx(Int idx) const { return mEdges[idx]; }
        inline UInt GetEdgeIdByIdx(Int idx) const { return mEdges[idx]; }
        inline const KiriVertex3Ptr &GetVertexByIdx(Int idx) const { return mVertices[idx]; }

        void SetIdx(UInt idx) { mIdx = idx; }

        void SetVisible(bool vis)
        {
            mVisible = vis;
        }

        inline const bool CheckConflict(const Vector3F &v) const
        {
            return (mNormal.dot(v) > (mNormal.dot(mVertices[0]->GetValue()) + MEpsilon<float>()));
        }

        inline constexpr bool IsVisibleFromBelow() const { return (mNormal.z < -MEpsilon<float>()); }

        //const KiriEdge3Ptr &GetEdge(const KiriVertex3Ptr &a, const KiriVertex3Ptr &b);

        void PrintFaceInfo();

        void GenerateEdges();

        void OrientFace(const KiriVertex3Ptr &orient);

        // void LinkEdge(const KiriEdge3Ptr &e);
        // void LinkFace(const SharedPtr<KiriFace3> &f);
        // void LinkFace(const SharedPtr<KiriFace3> &f, const KiriVertex3Ptr &a, const KiriVertex3Ptr &b);

    private:
        UInt mIdx;
        bool mVisible;
        KiriVertex3Ptr mVertices[3];
        Vector3F mNormal;
        // KiriEdge3Ptr mEdges[3];
        UInt mEdges[3];

        /*** 
         * @description: Check the relation between point v and face
         * @param {Vector3F} v
         * @return {true=blow false=above}
         */
        inline const bool CheckFaceDir(const Vector3F &v)
        {
            return (mNormal.dot(v) < mNormal.dot(mVertices[0]->GetValue()));
        }

        void ReverseFaceDir();
    };
    typedef SharedPtr<KiriFace3> KiriFace3Ptr;
} // namespace KIRI

#endif /* _KIRI_FACE3_H_ */