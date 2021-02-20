/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-19 18:18:39
 * @LastEditTime: 2020-11-04 23:25:34
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\geo\geo_object.h
 * @Reference:https://ttnghia.github.io ; Banana
 */

#ifndef _KIRI_GEO_OBJECT_H_
#define _KIRI_GEO_OBJECT_H_
#pragma once
#include <kiri_core/geo/geo_grid.h>

class KiriGeoObject
{
public:
    virtual String mName() = 0;
    virtual float signedDistance(const Vector3F &ppos0, bool bNegativeInside = true) const = 0;

    // auto getAABBMin() const { return transform(Vector3F(0.f)) - Vector3F(mUniformScale) * std::sqrt(Vector3F(1.f).sum()); }
    // auto getAABBMax() const { return transform(Vector3F(0.f)) + Vector3F(mUniformScale) * std::sqrt(Vector3F(1.f).sum()); }

    // auto getAABBMin() const { return transform(Vector3F(0.f)) - Vector3F(mUniformScale) * std::sqrt(Vector3F(1.f).sum()); }
    // auto getAABBMax() const { return transform(Vector3F(0.f)) + Vector3F(mUniformScale) * std::sqrt(Vector3F(1.f).sum()); }

    auto getAABBMin() const { return mAABBMin + mOffset; }
    auto getAABBMax() const { return mAABBMax + mOffset; }

    Vector3F transform(const Vector3F &ppos) const;

protected:
    bool mTransformed = false;
    float mUniformScale = 0.3f;
    Vector3F mAABBMin, mAABBMax;

    //FIXME force to get the correct model sampling
    Vector3F mOffset;
};

class KiriTriMeshObject : public KiriGeoObject
{
public:
    KiriTriMeshObject() = delete;
    KiriTriMeshObject(const String &meshFilePath, float sdfStep, Vector3F Offset, float BoxScale);

    String mName() override { return String("KiriTriMeshObject"); }
    virtual float signedDistance(const Vector3F &, bool bNegativeInside = true) const override;

    String &meshFile() { return mTriMeshFile; }
    float &sdfStep() { return mStep; }
    void computeSDF();

protected:
    void computeSDFMesh(const Vec_Vec3F &faces, const Vec_Vec3F &vertices, const Vector3F &origin, float CellSize,
                        float ni, float nj, float nk, Array3F &SDF, Int exactBand = 1);

    bool mSDFGenerated = false;
    String mTriMeshFile = String("");
    float mStep = 1.f / 256.f;
    float mBoxScale;

    Array3F mSDFData;
    KiriGeoGrid mGrid3D;
};

typedef SharedPtr<KiriTriMeshObject> KiriTriMeshObjectPtr;

#endif
