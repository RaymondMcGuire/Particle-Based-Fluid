/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 14:46:49
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-09-29 13:11:06
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\math\cuda_math_utils.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_MATH_UTILS_CUH_
#define _CUDA_MATH_UTILS_CUH_

#pragma once

#include <kiri_pbs_cuda/math/cuda_math_tensor.cuh>

struct rect3 {
  float3 origin;
  float3 size;
};

inline __host__ __device__ bool operator!=(float3 a, float3 b) {
  if (a.x == b.x && a.y == b.y && a.z == b.z)
    return false;
  else
    return true;
}

inline __host__ __device__ float3 roundf3(float3 v) {
  return make_float3(roundf(v.x), roundf(v.y), roundf(v.z));
}

inline __host__ __device__ float3 ones() { return make_float3(1.f); }

inline __host__ __device__ float3 zeros() { return make_float3(0.f); }

inline __host__ __device__ float3 ceilf3(float3 v) {
  return make_float3(ceilf(v.x), ceilf(v.y), ceilf(v.z));
}

inline __host__ __device__ float3 floorf3(float3 v) {
  return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}

inline __host__ __device__ int3 float3_to_int3(float3 v) {
  float3 v_floor = floorf3(v);
  return make_int3((int)v_floor.x, (int)v_floor.y, (int)v_floor.z);
}

/*! @brief Recalculates inertia tensor of a body after translation away from
 * (default) or towards its centroid.
 *
 * @param I inertia tensor in the original coordinates; it is assumed to be
 * upper-triangular (elements below the diagonal are ignored).
 * @param m mass of the body; if positive, translation is away from the
 * centroid; if negative, towards centroid.
 * @param off offset of the new origin from the original origin
 * @return inertia tensor in the new coordinate system; the matrix is symmetric.
 */
__host__ __device__ inline tensor3x3
inertiaTensorTranslate(tensor3x3 I, float m, float3 off) {
  return I + m * (dot(off, off) * make_identity() - make_tensor3x3(off, off));
}

__host__ __device__ inline float lengthSquared(float3 v) { return dot(v, v); }

template <typename T> __host__ __device__ inline int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

inline __host__ __device__ float3 sgn(float3 v) {
  return make_float3(sgn(v.x), sgn(v.y), sgn(v.z));
}

inline __host__ __device__ float fmaxf3(float3 f) {
  return fmaxf(fmaxf(f.x, f.y), f.z);
}

inline __host__ __device__ float fmaxf3(float f0, float f1, float f2) {
  return fmaxf(fmaxf(f0, f1), f2);
}

inline __host__ __device__ float3 fmaxf3(float3 f0, float3 f1, float3 f2) {
  return fmaxf(fmaxf(f0, f1), f2);
}

inline __host__ __device__ float3 fmaxf2(float3 f0, float3 f1) {
  return fmaxf(f0, f1);
}

inline __host__ __device__ float3 fminf2(float3 f0, float3 f1) {
  return fminf(f0, f1);
}

inline __host__ __device__ float fminf3(float3 f) {
  return fminf(fminf(f.x, f.y), f.z);
}

inline __host__ __device__ float fminf3(float f0, float f1, float f2) {
  return fminf(fminf(f0, f1), f2);
}

inline __host__ __device__ float3 fminf3(float3 f0, float3 f1, float3 f2) {
  return fminf(fminf(f0, f1), f2);
}

inline __host__ __device__ bool plane_aabb(const float3 n, const float d,
                                           const float3 c, const float3 l) {
  float e = dot(l, fabs(n));
  float s = dot(c, n) + d;
  if (s - e > 0.f)
    return false;
  if (s + e < 0.f)
    return false;
  return true;
}

inline __host__ __device__ bool edge_aabb(const float3 p0, const float3 p1,
                                          const float3 p2, const float3 f,
                                          const float3 l) {
  float3 e0 = make_float3(1.f, 0.f, 0.f), e1 = make_float3(0.f, 1.f, 0.f),
         e2 = make_float3(0.f, 0.f, 1.f);
  float3 a0 = cross(e0, f), a1 = cross(e1, f), a2 = cross(e2, f);

  float3 ap0 = make_float3(dot(a0, p0), dot(a0, p1), dot(a0, p2));
  float3 ap1 = make_float3(dot(a1, p0), dot(a1, p1), dot(a1, p2));
  float3 ap2 = make_float3(dot(a2, p0), dot(a2, p1), dot(a2, p2));

  float r0 = dot(l, fabs(a0));
  float r1 = dot(l, fabs(a1));
  float r2 = dot(l, fabs(a2));

  if (fminf3(ap0) > r0 || fmaxf3(ap0) < -r0 || fminf3(ap1) > r1 ||
      fmaxf3(ap1) < -r1 || fminf3(ap2) > r2 || fmaxf3(ap2) < -r2) {
    return true;
  }

  return false;
}

/**
 * @description:
 * @param n: normal, c: center, l: box half length
 * @return {*}
 */
inline __host__ __device__ bool triangle_aabb(const float3 p0, const float3 p1,
                                              const float3 p2, const float3 n,
                                              const float3 c, const float3 l) {
  float3 rp0, rp1, rp2;
  rp0 = p0 - c;
  rp1 = p1 - c;
  rp2 = p2 - c;

  // AABB Box
  float3 bmin = make_float3(-l.x, -l.y, -l.z),
         bmax = make_float3(l.x, l.y, l.z);

  // Triangle AABB
  float3 tmin = fminf3(rp0, rp1, rp2), tmax = fmaxf3(rp0, rp1, rp2);

  // Check Axis
  int overlap = 0;
  if (!(tmax.x < bmin.x || tmin.x > bmax.x)) {
    overlap++;
  }
  if (!(tmax.y < bmin.y || tmin.y > bmax.y)) {
    overlap++;
  }
  if (!(tmax.z < bmin.z || tmin.z > bmax.z)) {
    overlap++;
  }

  if (overlap < 3) {
    return false;
  }

  // Check Plane
  if (!plane_aabb(n, -dot(n, rp0), make_float3(0.f), l)) {
    return false;
  }

  // Check Box AABB With Edge

  float3 f0 = rp1 - rp0, f1 = rp2 - rp1, f2 = rp0 - rp2;

  if (edge_aabb(rp0, rp1, rp2, f0, l) || edge_aabb(rp0, rp1, rp2, f1, l) ||
      edge_aabb(rp0, rp1, rp2, f2, l)) {
    return false;
  }

  return true;
}

inline __host__ __device__ bool
polygon_intersects_cube(const float3 v0, const float3 v1, const float3 v2,
                        const float3 n, const float3 c) {
  bool intersect = false;

  if (triangle_aabb(v0, v1, v2, n, c, make_float3(0.5f))) {
    intersect = true;
  }

  return intersect;
}

inline __host__ __device__ bool isinside(const float3 pos, const float3 lower,
                                         const float3 upper) {
  if (pos.x < lower.x || pos.y < lower.y || pos.z < lower.z ||
      pos.x > upper.x || pos.y > upper.y || pos.z > upper.z)
    return false;

  return true;
}

inline __host__ __device__ void barycentric(float x, int &i, float &f,
                                            int i_low, int i_high) {
  auto s = floorf(x);
  i = static_cast<int>(s);

  if (i < i_low) {
    i = i_low;
    f = 0;
  } else if (i > i_high - 2) {
    i = i_high - 2;
    f = 1;
  } else {
    f = (float)(x - s);
  }
}

inline __host__ __device__ float bilerp(const float &v00, const float &v10,
                                        const float &v01, const float &v11,
                                        float fx, float fy) {
  return lerp(lerp(v00, v10, fx), lerp(v01, v11, fx), fy);
}

inline __host__ __device__ float trilerp(const float &v000, const float &v100,
                                         const float &v010, const float &v110,
                                         const float &v001, const float &v101,
                                         const float &v011, const float &v111,
                                         float fx, float fy, float fz) {
  return lerp(bilerp(v000, v100, v010, v110, fx, fy),
              bilerp(v001, v101, v011, v111, fx, fy), fz);
}

inline uint __host__ __device__ xyz2hash(int x, int y, int z, int3 grid_size) {
  return x + grid_size.x * (y + grid_size.y * z);
}

inline __host__ __device__ float
interpolate_value_linear(const float3 point, const float *grid_data,
                         const int3 grid_size) {
  int i, j, k;
  float fi, fj, fk;
  barycentric(point.x, i, fi, 0, grid_size.x);
  barycentric(point.y, j, fj, 0, grid_size.y);
  barycentric(point.z, k, fk, 0, grid_size.z);
  return trilerp(grid_data[xyz2hash(i, j, k, grid_size)],
                 grid_data[xyz2hash(i + 1, j, k, grid_size)],
                 grid_data[xyz2hash(i, j + 1, k, grid_size)],
                 grid_data[xyz2hash(i + 1, j + 1, k, grid_size)],
                 grid_data[xyz2hash(i, j, k + 1, grid_size)],
                 grid_data[xyz2hash(i + 1, j, k + 1, grid_size)],
                 grid_data[xyz2hash(i, j + 1, k + 1, grid_size)],
                 grid_data[xyz2hash(i + 1, j + 1, k + 1, grid_size)], fi, fj,
                 fk);
}

#endif /* _CUDA_MATH_UTILS_CUH_ */
