/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-19 18:18:39
 * @LastEditTime: 2021-02-20 19:37:37
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\model\model_tiny_obj_loader.h
 * @Reference:https://ttnghia.github.io ; Banana
 */

#ifndef _KIRI_MODEL_TINY_OBJ_LOADER_H_
#define _KIRI_MODEL_TINY_OBJ_LOADER_H_
#pragma once
#include <kiri_pch.h>

class KiriModelTinyObjLoader
{
public:
    KiriModelTinyObjLoader() { clearData(); }
    KiriModelTinyObjLoader(const String &name, const String &folder, const String &ext);

    bool Load(const String &filePath);
    void scaleToBox(float BoxScale);

    auto getMeshCenter() const
    {
        KIRI_ASSERT(mMeshReady);
        return float(0.5) * (mAABBMin + mAABBMax);
    }
    auto getNTriangles() const { return mNumTriangles; }
    const auto &getAABBMin() const
    {
        KIRI_ASSERT(mMeshReady);
        return mAABBMin;
    }
    const auto &getAABBMax() const
    {
        KIRI_ASSERT(mMeshReady);
        return mAABBMax;
    }

    const auto &getVertices() const
    {
        KIRI_ASSERT(mMeshReady);
        return mVertices;
    }
    const auto &getNormals() const
    {
        KIRI_ASSERT(mMeshReady);
        return mNormals;
    }
    const auto &getTexCoord2D() const
    {
        KIRI_ASSERT(mMeshReady);
        return mTexCoord2D;
    }

    const auto &getFaces() const
    {
        KIRI_ASSERT(mMeshReady);
        return mFaces;
    }
    const auto &getFaceVertices() const
    {
        KIRI_ASSERT(mMeshReady);
        return mFaceVertices;
    }
    const auto &getFaceVertexNormals() const
    {
        KIRI_ASSERT(mMeshReady);
        return mFaceVertexNormals;
    }
    const auto &getFaceVertexColors() const
    {
        KIRI_ASSERT(mMeshReady);
        return mFaceVertexColors;
    }
    const auto &getFaceVTexCoord2D() const
    {
        KIRI_ASSERT(mMeshReady);
        return mFaceVertexTexCoord2D;
    }

    auto getNFaces() const noexcept
    {
        KIRI_ASSERT(mMeshReady);
        return (mFaces.size() / 3);
    }
    auto getNVertices() const noexcept
    {
        KIRI_ASSERT(mMeshReady);
        return (mVertices.size() / 3);
    }
    auto getNFaceVertices() const noexcept
    {
        KIRI_ASSERT(mMeshReady);
        return (mFaceVertices.size() / 3);
    }

private:
    void computeFaceVertexData();
    void clearData();

    String mName;
    String mExtension;
    String mFolder;

    bool mMeshReady = false;
    UInt mNumTriangles = 0;

    Vector3F mAABBMin;
    Vector3F mAABBMax;

    //TODO change type to uint
    Vec_Float mFaces;
    Vec_Float mVertices;
    Vec_Float mNormals;
    Vec_Float mTexCoord2D;
    Vec_Float mFaceVertices;
    Vec_Float mFaceVertexNormals;
    Vec_Float mFaceVertexColors;
    Vec_Float mFaceVertexTexCoord2D;
};

typedef SharedPtr<KiriModelTinyObjLoader> KiriModelTinyObjLoaderPtr;

#endif
