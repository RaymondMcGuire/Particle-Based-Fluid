/*** 
 * @Author: Xu.WANG
 * @Date: 2021-03-27 01:49:01
 * @LastEditTime: 2021-06-04 16:37:36
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri2D\Kiri2d\include\kiri_core\data\vertex3.h
 */

#ifndef _KIRI_VERTEX3_H_
#define _KIRI_VERTEX3_H_

#pragma once

#include <kiri_pch.h>

namespace KIRI
{
    class KiriVertex3
    {
    public:
        explicit KiriVertex3() : KiriVertex3(Vector3F(0.f)) {}

        explicit KiriVertex3(Vector3F v) : KiriVertex3(-1, v) {}

        explicit KiriVertex3(UInt idx, Vector3F v)
        {
            mIdx = idx;
            mValue = v;
        }

        virtual ~KiriVertex3() noexcept {}

        void SetIdx(UInt idx) { mIdx = idx; }
        constexpr UInt GetIdx() { return mIdx; }

        void SetValue(const Vector3F &v) { mValue = v; }
        const Vector3F &GetValue() { return mValue; }

        const bool LinearDependent(const Vector3F &v);

        inline const bool IsEqual(const SharedPtr<KiriVertex3> &vert) const
        {
            return (mValue.x == vert->GetValue().x && mValue.y == vert->GetValue().y && mValue.z == vert->GetValue().z && mIdx == vert->GetIdx());
        }

        void Print()
        {
            KIRI_LOG_DEBUG("vertex idx={0}", mIdx);
            KIRI_LOG_DEBUG("vertex value=({0},{1},{2})",
                           mValue.x, mValue.y, mValue.z);
        }

    private:
        Vector3F mValue;
        UInt mIdx;
    };
    typedef SharedPtr<KiriVertex3> KiriVertex3Ptr;
} // namespace KIRI

#endif /* _KIRI_VERTEX3_H_ */