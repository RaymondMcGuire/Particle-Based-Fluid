/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-19 18:18:39
 * @LastEditTime: 2020-10-22 14:15:19
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\geo\geo_helper.cpp
 * @Reference:https://ttnghia.github.io ; Banana
 */

#include <kiri_core/geo/geo_math_helper.h>
#include <kiri_core/geo/geo_helper.h>

namespace KIRI
{

    // find distance x0 is from segment x1-x2
    float point_segment_distance(const Vector3F &x0, const Vector3F &x1, const Vector3F &x2)
    {
        Vector3F dx(x2 - x1);

        float m2 = dx.lengthSquared();
        // find parameter value of closest point on segment
        float s12 = dx.dot(x2 - x0) / m2;

        if (s12 < 0)
        {
            s12 = 0;
        }
        else if (s12 > 1)
        {
            s12 = 1;
        }

        return (x0 - s12 * x1 + (1 - s12) * x2).length();
    }

    // find distance x0 is from triangle x1-x2-x3
    float point_triangle_distance(const Vector3F &x0, const Vector3F &x1, const Vector3F &x2, const Vector3F &x3)
    {
        // first find barycentric coordinates of closest point on infinite plane
        Vector3F x13(x1 - x3), x23(x2 - x3), x03(x0 - x3);
        float m13 = x13.lengthSquared(), m23 = x23.lengthSquared(), d = x13.dot(x23);

        float invdet = 1.f / fmax(m13 * m23 - d * d, 1e-30f);
        float a = x13.dot(x03), b = x23.dot(x03);

        // the barycentric coordinates themselves
        float w23 = invdet * (m23 * a - d * b);
        float w31 = invdet * (m13 * b - d * a);
        float w12 = 1 - w23 - w31;

        if (w23 >= 0 && w31 >= 0 && w12 >= 0)
        { // if we're inside the triangle
            return (x0 - w23 * x1 + w31 * x2 + w12 * x3).length();
        }
        else
        { // we have to clamp to one of the edges
            if (w23 > 0)
            { // this rules out edge 2-3 for us
                return std::min(point_segment_distance(x0, x1, x2), point_segment_distance(x0, x1, x3));
            }
            else if (w31 > 0)
            { // this rules out edge 1-3
                return std::min(point_segment_distance(x0, x1, x2), point_segment_distance(x0, x2, x3));
            }
            else
            { // w12 must be >0, ruling out edge 1-2
                return std::min(point_segment_distance(x0, x1, x3), point_segment_distance(x0, x2, x3));
            }
        }
    }

    void check_neighbour(const Vec_Vec3F &tri, const Vec_Vec3F &x, Array3F &phi, Array3UI &closest_tri,
                         const Vector3F &gx,
                         Int i0, Int j0, Int k0,
                         Int i1, Int j1, Int k1)
    {
        if (closest_tri(i1, j1, k1) != 0xffffffff)
        {
            UInt p = tri[closest_tri(i1, j1, k1)][0];
            UInt q = tri[closest_tri(i1, j1, k1)][1];
            UInt r = tri[closest_tri(i1, j1, k1)][2];

            float d = point_triangle_distance(gx, x[p], x[q], x[r]);

            if (d < phi(i0, j0, k0))
            {
                phi(i0, j0, k0) = d;
                closest_tri(i0, j0, k0) = closest_tri(i1, j1, k1);
            }
        }
    }

    void sweep(const Vec_Vec3F &tri, const Vec_Vec3F &x,
               Array3F &phi, Array3UI &closest_tri, const Vector3F &origin, float dx,
               Int di, Int dj, Int dk)
    {
        Int i0, i1;
        Int j0, j1;
        Int k0, k1;

        if (di > 0)
        {
            i0 = 1;
            i1 = static_cast<Int>(phi.size()[0]);
        }
        else
        {
            i0 = static_cast<Int>(phi.size()[0]) - 2;
            i1 = -1;
        }

        if (dj > 0)
        {
            j0 = 1;
            j1 = static_cast<Int>(phi.size()[1]);
        }
        else
        {
            j0 = static_cast<Int>(phi.size()[1]) - 2;
            j1 = -1;
        }

        if (dk > 0)
        {
            k0 = 1;
            k1 = static_cast<Int>(phi.size()[2]);
        }
        else
        {
            k0 = static_cast<Int>(phi.size()[2]) - 2;
            k1 = -1;
        }

        //    Scheduler::parallel_for<Int>(i0, i1 + 1, j0, j1 + 1, k0, k1 + 1,
        //                                       [&](Int i, Int j, Int k)

        for (Int k = k0; k != k1; k += dk)
        {
            for (Int j = j0; j != j1; j += dj)
            {
                for (Int i = i0; i != i1; i += di)
                {
                    Vector3F gx = Vector3F(i, j, k) * dx + origin;

                    check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i - di, j, k);
                    check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i, j - dj, k);
                    check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i - di, j - dj, k);
                    check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i, j, k - dk);
                    check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i - di, j, k - dk);
                    check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i, j - dj, k - dk);
                    check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i - di, j - dj, k - dk);
                }
            }
        }
    }

    // calculate twice signed area of triangle (0,0)-(x1,y1)-(x2,y2)
    // return an SOS-determined sign (-1, +1, or 0 only if it's a truly degenerate triangle)
    Int orientation(float x1, float y1, float x2, float y2, float &twice_signed_area)
    {
        twice_signed_area = y1 * x2 - x1 * y2;

        if (twice_signed_area > 0)
        {
            return 1;
        }
        else if (twice_signed_area < 0)
        {
            return -1;
        }
        else if (y2 > y1)
        {
            return 1;
        }
        else if (y2 < y1)
        {
            return -1;
        }
        else if (x1 > x2)
        {
            return 1;
        }
        else if (x1 < x2)
        {
            return -1;
        }
        else
        {
            return 0; // only true when x1==x2 and y1==y2
        }
    }

    bool point_in_triangle_2d(float x0, float y0,
                              float x1, float y1, float x2, float y2, float x3, float y3,
                              float &a, float &b, float &c)
    {
        x1 -= x0;
        x2 -= x0;
        x3 -= x0;
        y1 -= y0;
        y2 -= y0;
        y3 -= y0;
        Int signa = orientation(x2, y2, x3, y3, a);

        if (signa == 0)
        {
            return false;
        }

        Int signb = orientation(x3, y3, x1, y1, b);

        if (signb != signa)
        {
            return false;
        }

        Int signc = orientation(x1, y1, x2, y2, c);

        if (signc != signa)
        {
            return false;
        }

        float sum = a + b + c;
        KIRI_ASSERT(sum != 0); // if the SOS signs match and are nonkero, there's no way all of a, b, and c are zero.
        a /= sum;
        b /= sum;
        c /= sum;
        return true;
    }

    float interpolateValueLinear(const Vector3F &point, const Array3F &grid)
    {
        Int i, j, k;
        float fi, fj, fk;
        get_barycentric(point[0], i, fi, 0, static_cast<Int>(grid.size()[0]));
        get_barycentric(point[1], j, fj, 0, static_cast<Int>(grid.size()[1]));
        get_barycentric(point[2], k, fk, 0, static_cast<Int>(grid.size()[2]));
        return trilerp(
            grid(i, j, k), grid(i + 1, j, k), grid(i, j + 1, k), grid(i + 1, j + 1, k),
            grid(i, j, k + 1), grid(i + 1, j, k + 1), grid(i, j + 1, k + 1), grid(i + 1, j + 1, k + 1),
            fi, fj, fk);
    }

} //  namespace KIRI