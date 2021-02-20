/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-21 19:31:19
 * @LastEditTime: 2020-10-21 20:00:59
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\geo\geo_grid.cpp
 */

#include <kiri_core/geo/geo_grid.h>

void KiriGeoGrid::SetGrid(const Vector3F &bMin, const Vector3F &bMax, float CellSize)
{
    mBMin = bMin;
    mBMax = bMax;
    SetCellSize(CellSize);
}

void KiriGeoGrid::SetCellSize(float CellSize)
{
    KIRI_ASSERT(CellSize > 0);
    mCellSize = CellSize;
    mInvCellSize = 1.f / mCellSize;
    mHalfCellSize = 0.5f * mCellSize;
    mCellSizeSqr = mCellSize * mCellSize;
    mNTotalCells = 1;

    for (Int i = 0; i < mNCells.size(); ++i)
    {
        mNCells[i] = static_cast<UInt>(ceil((mBMax[i] - mBMin[i]) / mCellSize));
        mNNodes[i] = mNCells[i] + 1u;

        mNTotalCells *= mNCells[i];
        mNTotalNodes *= mNNodes[i];
    }

    // KIRI_LOG_INFO("Grid CellSize:{0:f}", CellSize);
    // KIRI_LOG_INFO("Grid GridCellSize:{0:d}", mNCells.size());
    // KIRI_LOG_INFO("Grid Cell:({0:f},{1:f},{2:f})", mNCells.x, mNCells.y, mNCells.z);
    // KIRI_LOG_INFO("Grid Node:({0:f},{1:f},{2:f})", mNNodes.x, mNNodes.y, mNNodes.z);
    // KIRI_LOG_INFO("Grid Total Cell Num:{0:d}", mNTotalCells);
    // KIRI_LOG_INFO("Grid Total Node Num::{0:d}", mNTotalNodes);

    mbCellIdxNeedResize = true;
}