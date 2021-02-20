/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-19 18:18:39
 * @LastEditTime: 2021-02-20 19:45:20
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\geo\geo_grid.h
 * @Reference:https://ttnghia.github.io ; Banana
 */

#ifndef _KIRI_GEO_GRID_H_
#define _KIRI_GEO_GRID_H_
#pragma once
#include <kiri_pch.h>

class KiriGeoGrid
{
public:
    KiriGeoGrid() = default;
    KiriGeoGrid(const Vector3F &bMin, const Vector3F &bMax, float CellSize) : mBMin(bMin), mBMax(bMax) { SetCellSize(CellSize); }

    void SetGrid(const Vector3F &bMin, const Vector3F &bMax, float CellSize);
    void SetCellSize(float CellSize);

    const auto &getBMin() const noexcept { return mBMin; }
    const auto &getBMax() const noexcept { return mBMax; }

    const auto &getNCells() const noexcept { return mNCells; }
    const auto &getNNodes() const noexcept { return mNNodes; }
    auto getNTotalCells() const noexcept { return mNTotalCells; }
    auto getNTotalNodes() const noexcept { return mNTotalNodes; }

    auto getCellSize() const noexcept { return mCellSize; }
    auto getInvCellSize() const noexcept { return mInvCellSize; }
    auto getHalfCellSize() const noexcept { return mHalfCellSize; }
    auto getCellSizeSquared() const noexcept { return mCellSizeSqr; }

    // Particle processing
    auto getGridCoordinate(const Vector3F &ppos) const { return (ppos - mBMin) / mCellSize; }

protected:
    Vector3F mBMin;
    Vector3F mBMax;

    Vector3F mNCells;
    Vector3F mNNodes;

    UInt mNTotalCells = 1u;
    UInt mNTotalNodes = 1u;
    float mCellSize = 1.f;
    float mInvCellSize = 1.f;
    float mHalfCellSize = 0.5f;
    float mCellSizeSqr = 1.f;

    bool mbCellIdxNeedResize = false; // to track and resize the mCellParticleIdx array
};

#endif
