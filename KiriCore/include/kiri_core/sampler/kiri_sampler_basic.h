/*** 
 * @Author: Pierre-Luc Manteaux
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2020-12-07 00:02:29
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\sampler\kiri_sampler_basic.h
 * @Reference: https://github.com/manteapi/hokusai
 */

#ifndef _KIRI_SAMPLER_BASIC_H_
#define _KIRI_SAMPLER_BASIC_H_
#pragma once
#include <kiri_core/mesh/mesh_triangle.h>

enum SamplerBasicType
{
    CubeSampler = 1,
    BoxSampler = 2
};

class KiriSamplerBasic
{
public:
    KiriSamplerBasic();

    Array1Vec3F GetCylinderSampling(const Vector3F center, float height, float baseRadius, float spacingX, float spacingY);
    Array1Vec3F GetConeSampling(const Vector3F center, float height, float stopHeight, float baseRadius, float spacingX, float spacingY);
    Array1Vec3F GetTorusSampling(const Vector3F center, float tubeRadius, float innerRadius, float spacingX, float spacingY);
    Array1Vec3F GetHemiSphereSampling(const Vector3F center, float radius, float spacingX, float spacingY);
    Array1Vec3F GetCapsuleSampling(const Vector3F center, float radius, float height, float spacingX, float spacingY);
    Array1Vec3F GetEllipsoidSampling(const Vector3F center, float axis_1, float axis_2, float axis_3, float spacingX, float spacingY);
    Array1Vec3F GetSphereSampling(const Vector3F center, float radius, float spacingX, float spacingY);
    Array1Vec3F GetCubeSampling(Vector3F center, Vector3F sides, float spacing);
    Array1Vec3F GetBoxSampling(Vector3F center, Vector3F sides, float spacing);
    Array1Vec3F GetDiskSampling(Vector3F center, float radius, float spacing);

    Array1Vec4F GetCubeSamplingWithRadius(Vector3F center, Vector3F sides, float spacing);

    //model sampling
    bool LineLineIntersect(const Vector3F &p1, const Vector3F &p2, const Vector3F &p3, const Vector3F &p4, Vector3F &pa, Vector3F &pb, float &mua, float &mub);
    bool AkinciTriangleSampling(const Vector3F &Point1, const Vector3F &Point2, const Vector3F &Point3, const float &Radius, Array1Vec3F &Samples);
    bool AkinciEdgeSampling(const Vector3F &Point1, const Vector3F &Point2, const float &Radius, Array1Vec3F &Samples);
    bool AkinciMeshSampling(const KiriMeshTriangle *Mesh, const float &Radius, Array1Vec4F &Samples);

private:
    Array1Vec3F mPoints;
};

typedef SharedPtr<KiriSamplerBasic> KiriSamplerBasicPtr;

#endif