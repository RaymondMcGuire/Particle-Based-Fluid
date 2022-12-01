/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 15:26:45
 * @FilePath: \Kiri\renderer\src\kiri_core\model\model_tiny_obj_loader.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#include <kiri_core/model/model_tiny_obj_loader.h>
#include <root_directory.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

KiriModelTinyObjLoader::KiriModelTinyObjLoader(const String &name, const String &folder, const String &ext, const String &fileName)
    : mName(fileName), mExtension(ext), mFolder(folder)
{
    if (mName == "")
        mName = name;

    String filePath = String(DB_PBR_PATH) + mFolder + "/" + name + "/" + mName + mExtension;

    if (RELEASE && PUBLISH)
    {
        // filePath = String(DB_PBR_PATH) + mFolder + "/" + mName + "/" + mName + mExtension;
        filePath = "./resources/" + mFolder + "/" + name + "/" + mName + mExtension;
    }
    KIRI_LOG_INFO("Tiny Obj Loader Model Path={0:s}", filePath);

    ClearData();
    mMeshReady = Load(filePath);

    KIRI_LOG_INFO("Tiny Obj Loader Status={0:s}", mMeshReady ? "True" : "False");
    KIRI_LOG_INFO("Vertex Number={0:d}", mVertices.size());

    if (mMeshReady)
        ComputeFaceVertexData();
};

void KiriModelTinyObjLoader::ClearData()
{
    mMeshReady = false;
    mNumTriangles = 0;

    mAABBMin = Vector3F(1e10f);
    mAABBMax = Vector3F(-1e10f);

    mVertices.resize(0);
    mNormals.resize(0);

    mFaces.resize(0);
    mFaceVertices.resize(0);
    mFaceVertexNormals.resize(0);
    mFaceVertexColors.resize(0);
    mFaceVertexTexCoord2D.resize(0);
}

bool KiriModelTinyObjLoader::Load(const String &filePath)
{
    Vector<tinyobj::shape_t> obj_shapes;
    Vector<tinyobj::material_t> obj_materials;
    tinyobj::attrib_t attrib;

    String warnStr, errStr;
    bool result = tinyobj::LoadObj(&attrib, &obj_shapes, &obj_materials, &warnStr, &errStr, filePath.c_str());

    if (!errStr.empty())
    {
        std::cerr << "tinyobj: " << errStr << std::endl;
    }

    if (!result)
    {
        std::cerr << "Failed to load " << filePath << std::endl;
        return false;
    }

    mVertices = attrib.vertices;
    mNormals = attrib.normals;
    mTexCoord2D = attrib.texcoords;

    ////////////////////////////////////////////////////////////////////////////////
    // => convert data
    for (size_t s = 0; s < obj_shapes.size(); s++)
    {
        for (size_t f = 0; f < obj_shapes[s].mesh.indices.size() / 3; ++f)
        {
            ++mNumTriangles;

            tinyobj::index_t idx0 = obj_shapes[s].mesh.indices[3 * f + 0];
            tinyobj::index_t idx1 = obj_shapes[s].mesh.indices[3 * f + 1];
            tinyobj::index_t idx2 = obj_shapes[s].mesh.indices[3 * f + 2];

            Int v0 = idx0.vertex_index;
            Int v1 = idx1.vertex_index;
            Int v2 = idx2.vertex_index;
            assert(v0 >= 0);
            assert(v1 >= 0);
            assert(v2 >= 0);

            // KIRI_LOG_DEBUG("f={0}/{1}/{2} {3}/{4}/{5} {6}/{7}/{8}",
            //                idx0.vertex_index, idx0.texcoord_index, idx0.normal_index,
            //                idx1.vertex_index, idx1.texcoord_index, idx1.normal_index,
            //                idx2.vertex_index, idx2.texcoord_index, idx2.normal_index);

            mFaces.push_back(static_cast<UInt>(v0));
            mFaces.push_back(static_cast<UInt>(v1));
            mFaces.push_back(static_cast<UInt>(v2));

            Vector3F v[3];
            for (Int k = 0; k < 3; ++k)
            {
                v[0][k] = mVertices[3 * v0 + k];
                v[1][k] = mVertices[3 * v1 + k];
                v[2][k] = mVertices[3 * v2 + k];

                mAABBMin[k] = std::min(v[0][k], mAABBMin[k]);
                mAABBMin[k] = std::min(v[1][k], mAABBMin[k]);
                mAABBMin[k] = std::min(v[2][k], mAABBMin[k]);

                mAABBMax[k] = std::max(v[0][k], mAABBMax[k]);
                mAABBMax[k] = std::max(v[1][k], mAABBMax[k]);
                mAABBMax[k] = std::max(v[2][k], mAABBMax[k]);
            }

            for (Int k = 0; k < 3; ++k)
            {
                mFaceVertices.push_back(v[k][0]);
                mFaceVertices.push_back(v[k][1]);
                mFaceVertices.push_back(v[k][2]);
            }

            if (attrib.normals.size() > 0 && idx0.normal_index >= 0)
            {
                Vector3F n[3];
                Int n0 = idx0.normal_index;
                Int n1 = idx1.normal_index;
                Int n2 = idx2.normal_index;
                assert(n0 >= 0);
                assert(n1 >= 0);
                assert(n2 >= 0);

                for (Int k = 0; k < 3; ++k)
                {
                    n[0][k] = attrib.normals[3 * n0 + k];
                    n[1][k] = attrib.normals[3 * n1 + k];
                    n[2][k] = attrib.normals[3 * n2 + k];
                }

                for (Int k = 0; k < 3; ++k)
                {
                    mFaceVertexNormals.push_back(n[k][0]);
                    mFaceVertexNormals.push_back(n[k][1]);
                    mFaceVertexNormals.push_back(n[k][2]);
                }
            }

            if (attrib.texcoords.size() > 0 && idx0.texcoord_index >= 0)
            {
                Vector3F tex[3];
                Int t0 = idx0.texcoord_index;
                Int t1 = idx1.texcoord_index;
                Int t2 = idx2.texcoord_index;
                assert(t0 >= 0);
                assert(t1 >= 0);
                assert(t2 >= 0);

                for (Int k = 0; k < 2; ++k)
                {
                    tex[0][k] = attrib.texcoords[2 * t0 + k];
                    tex[1][k] = attrib.texcoords[2 * t1 + k];
                    tex[2][k] = attrib.texcoords[2 * t1 + k];
                }

                for (Int k = 0; k < 3; ++k)
                {
                    mFaceVertexTexCoord2D.push_back(tex[k][0]);
                    mFaceVertexTexCoord2D.push_back(tex[k][1]);
                }
            }
        }
    }
    return result;
}

void KiriModelTinyObjLoader::ComputeFaceVertexData()
{
    if (mFaceVertexNormals.size() != mFaceVertices.size())
    {
        mFaceVertexNormals.assign(mFaceVertices.size(), 0);
        mFaceVertexColors.assign(mFaceVertices.size(), 0);
        mNormals.assign(mVertices.size(), 0);

        for (UInt f = 0, f_end = GetNFaces(); f < f_end; ++f)
        {
            // Get index of vertices for the current face
            UInt v0 = mFaces[3 * f];
            UInt v1 = mFaces[3 * f + 1];
            UInt v2 = mFaces[3 * f + 2];

            mIndices.emplace_back(3 * f);
            mIndices.emplace_back(3 * f + 1);
            mIndices.emplace_back(3 * f + 2);

            Vector3F v[3];
            for (Int k = 0; k < 3; ++k)
            {
                v[0][k] = mVertices[3 * v0 + k];
                v[1][k] = mVertices[3 * v1 + k];
                v[2][k] = mVertices[3 * v2 + k];
            }
            pos.emplace_back(Vector3D(v[0].x, v[0].y, v[0].z));
            pos.emplace_back(Vector3D(v[1].x, v[1].y, v[1].z));
            pos.emplace_back(Vector3D(v[2].x, v[2].y, v[2].z));

            Vector3F faceNormal = ((v[1] - v[0]).cross(v[2] - v[0])).normalized();

            for (Int k = 0; k < 3; ++k)
            {
                mNormals[v0 * 3 + k] += faceNormal[k];
                mNormals[v1 * 3 + k] += faceNormal[k];
                mNormals[v2 * 3 + k] += faceNormal[k];
            }

            mNormal.emplace_back(Vector3D(mNormals[v0 * 3 + 0], mNormals[v0 * 3 + 1], mNormals[v0 * 3 + 2]));
            mNormal.emplace_back(Vector3D(mNormals[v1 * 3 + 0], mNormals[v1 * 3 + 1], mNormals[v1 * 3 + 2]));
            mNormal.emplace_back(Vector3D(mNormals[v2 * 3 + 0], mNormals[v2 * 3 + 1], mNormals[v2 * 3 + 2]));
        }

        for (size_t f = 0, f_end = GetNFaces(); f < f_end; ++f)
        {
            UInt v0 = mFaces[3 * f];
            UInt v1 = mFaces[3 * f + 1];
            UInt v2 = mFaces[3 * f + 2];

            Vec_Vec3F fNormals(3, Vector3F(0.f));
            for (Int k = 0; k < 3; ++k)
            {
                fNormals[0][k] = mNormals[3 * v0 + k];
                fNormals[1][k] = mNormals[3 * v1 + k];
                fNormals[2][k] = mNormals[3 * v2 + k];
            }

            for (Int k = 0; k < 3; ++k)
            {
                for (Int l = 0; l < 3; ++l)
                {
                    mFaceVertexNormals[9 * f + 3 * k + l] = fNormals[k][l];
                    mFaceVertexColors[9 * f + 3 * k + l] = fNormals[k][l];
                }
            }
        }
    }
}

void KiriModelTinyObjLoader::ScaleToBox(float BoxScale)
{
    Vector3F diff = mAABBMax - mAABBMin;
    float maxSize = fmaxf(fmaxf(std::abs(diff[0]), std::abs(diff[1])), std::abs(diff[2]));
    float Scale = 2.0f / maxSize * BoxScale;

    // multiply all vertices by Scale to make the mesh having max(w, h, d) = 1
    mAABBMin = mAABBMin * Scale;
    mAABBMax = mAABBMax * Scale;

    // expand the bounding box
    Vector3F meshCenter = (mAABBMax + mAABBMin) * 0.5f;
    auto cmin = mAABBMin - meshCenter;
    auto cmax = mAABBMax - meshCenter;

    mAABBMin = meshCenter + cmin.normalized() * cmin.length();
    mAABBMax = meshCenter + cmax.normalized() * cmax.length();

    // to move the mesh center to origin
    mAABBMin = mAABBMin - meshCenter;
    mAABBMax = mAABBMax - meshCenter;

    Vector3F *vertexPtr = reinterpret_cast<Vector3F *>(mVertices.data());
    // KIRI_LOG_INFO("ScaleToBox, VertexPtrValue[0]=({0:f},{1:f},{2:f})", vertexPtr[0].x, vertexPtr[0].y, vertexPtr[0].z);
    for (size_t i = 0, iend = mVertices.size() / 3; i < iend; ++i)
    {
        vertexPtr[i] = vertexPtr[i] * Scale;
        vertexPtr[i] = vertexPtr[i] - meshCenter;
    }

    Vector3F *faceVertexPtr = reinterpret_cast<Vector3F *>(mFaceVertices.data());
    for (size_t i = 0, iend = mFaceVertices.size() / 3; i < iend; ++i)
    {
        faceVertexPtr[i] = faceVertexPtr[i] * Scale;
        faceVertexPtr[i] = faceVertexPtr[i] - meshCenter;
    }
}

void KiriModelTinyObjLoader::Normalize()
{
    // translate vertices to original
    Vector3F trans(-mAABBMin);
    KIRI_LOG_INFO("Normalize, trans=({0},{1},{2})", trans.x, trans.y, trans.z);
    Vector3F *faceVertexPtr = reinterpret_cast<Vector3F *>(mFaceVertices.data());

    float scale = 1.f / (mAABBMax[0] - mAABBMin[0]);
    for (size_t k = 1; k < 3; k++)
    {
        scale = std::min((1.f / (mAABBMax[k] - mAABBMin[k])), scale);
    }

    mAABBMin = Vector3F(1e10f);
    mAABBMax = Vector3F(-1e10f);
    for (size_t i = 0, iend = mFaceVertices.size() / 3; i < iend; ++i)
    {
        faceVertexPtr[i] += trans;
        faceVertexPtr[i] *= scale;

        for (size_t k = 0; k < 3; k++)
        {
            mAABBMin[k] = std::min(faceVertexPtr[i][k], mAABBMin[k]);
            mAABBMin[k] = std::min(faceVertexPtr[i][k], mAABBMin[k]);
            mAABBMin[k] = std::min(faceVertexPtr[i][k], mAABBMin[k]);

            mAABBMax[k] = std::max(faceVertexPtr[i][k], mAABBMax[k]);
            mAABBMax[k] = std::max(faceVertexPtr[i][k], mAABBMax[k]);
            mAABBMax[k] = std::max(faceVertexPtr[i][k], mAABBMax[k]);
        }
    }
}