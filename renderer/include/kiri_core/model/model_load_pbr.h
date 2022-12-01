/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-03-24 20:58:37
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\model\model_load_pbr.h
 */

#ifndef _KIRI_MODEL_LOAD_PBR_H_
#define _KIRI_MODEL_LOAD_PBR_H_
#pragma once
#include <kiri_define.h>
//3rd-party library
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <root_directory.h>
#include <kiri_core/model/model_load.h>
#include <kiri_core/mesh/mesh_pbr.h>
#include <kiri_core/texture/pbr_texture.h>

#include <kiri_core/camera/camera.h>

class KiriModelLoadPBR : public KiriModelLoad
{
public:
    KiriModelLoadPBR(String Name, bool Gamma = false, String Folder = "models", String Extension = ".fbx", String TextureExtension = ".tga")
        : mGamma(Gamma), mFolder(Folder), mExtension(Extension), mName(Name)
    {
        mPath = String(DB_PBR_PATH) + mFolder + "/" + mName + "/" + mName + mExtension;
        if (RELEASE && PUBLISH)
        {
            // mPath = String(DB_PBR_PATH) + mFolder + "/" + mName + "/" + mName + mExtension;
            mPath = "./resources/" + mFolder + "/" + mName + "/" + mName + mExtension;
        }
        mPBRTexture = std::make_shared<KiriPBRTexture>(mName, mGamma, mFolder, TextureExtension);
        mPBRTexture->Load();

        mAABBMin = Vector3F(1e10f);
        mAABBMax = Vector3F(-1e10f);
    }

    void Draw() override;
    void Load(Vector3F Offset = Vector3F(0.f), Vector3F Scale = Vector3F(1.f));

    inline KiriPBRTexturePtr GetPBRTexture() const { return mPBRTexture; }

    inline Array1<VertexFull> Vertexs() const { return mVertexs; }
    inline Array1<UInt> Indices() const { return mIndices; }

    inline Vector3F AABBMin() const { return mAABBMin; }
    inline Vector3F AABBMax() const { return mAABBMax; }
    inline Vector3F BBoxMin() const { return mBBoxMin; }
    inline Vector3F BBoxMax() const { return mBBoxMax; }

    Array1<Vector2F> Project2Screen(KIRI::KiriCameraPtr camera);

    void GetEdges(Array1<std::pair<Int, Int>> &Edges) const;
    bool AkinciMeshSampling(const float &Radius, Array1Vec4F &Samples);
    bool AkinciEdgeSampling(const Vector3F &Point1, const Vector3F &Point2, const float &Radius, Array1Vec3F &Samples);
    bool AkinciTriangleSampling(const Vector3F &Point1, const Vector3F &Point2, const Vector3F &Point3, const float &Radius, Array1Vec3F &Samples);
    bool LineLineIntersect(const Vector3F &p1, const Vector3F &p2, const Vector3F &p3, const Vector3F &p4, Vector3F &pa, Vector3F &pb, float &mua, float &mub);

private:
    void ProcessNode(aiNode *, const aiScene *);
    KiriMeshPBRPtr ProcessMesh(aiMesh *, const aiScene *);
    void ReMakeVertexPosition(Vector3F Offset, Vector3F Scale, bool Normalize = false);

    String mName, mExtension, mFolder, mPath;
    bool mGamma;

    KiriPBRTexturePtr mPBRTexture;
    Array1<KiriMeshPBRPtr> mMeshes;

    Vector3F mOffset, mScale;
    Vector3F mAABBMin, mAABBMax, mBBoxMin, mBBoxMax;

    Array1<VertexFull> mVertexs;
    Array1<UInt> mIndices;
    Array1Vec3F mTriangles;
};

typedef SharedPtr<KiriModelLoadPBR> KiriModelLoadPBRPtr;
#endif