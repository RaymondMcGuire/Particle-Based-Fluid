/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-19 18:18:39
 * @LastEditTime: 2020-10-22 14:15:10
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\geo\geo_helper.h
 * @Reference:https://ttnghia.github.io ; Banana
 */

#ifndef _KIRI_GEO_HELPER_H_
#define _KIRI_GEO_HELPER_H_
#pragma once
#include <kiri_pch.h>

namespace KIRI
{

    float point_segment_distance(const Vector3F &x0, const Vector3F &x1, const Vector3F &x2);

    float point_triangle_distance(const Vector3F &x0, const Vector3F &x1, const Vector3F &x2, const Vector3F &x3);

    void check_neighbour(const Vec_Vec3F &tri, const Vec_Vec3F &x, Array3F &phi, Array3UI &closest_tri,
                         const Vector3F &gx,
                         Int i0, Int j0, Int k0,
                         Int i1, Int j1, Int k1);

    void sweep(const Vec_Vec3F &tri, const Vec_Vec3F &x,
               Array3F &phi, Array3UI &closest_tri, const Vector3F &origin, float dx,
               Int di, Int dj, Int dk);

    // robust test of (x0,y0) in the triangle (x1,y1)-(x2,y2)-(x3,y3)
    // if true is returned, the barycentric coordinates are set in a,b,c.
    bool point_in_triangle_2d(float x0, float y0,
                              float x1, float y1, float x2, float y2, float x3, float y3,
                              float &a, float &b, float &c);

    float interpolateValueLinear(const Vector3F &point, const Array3F &grid);

} // namespace KIRI
#endif