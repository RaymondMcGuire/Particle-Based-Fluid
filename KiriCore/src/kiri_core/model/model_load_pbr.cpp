/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2020-12-30 22:06:18
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\model\model_load_pbr.cpp
 */

#include <kiri_core/model/model_load_pbr.h>

void KiriModelLoadPBR::Draw()
{
    KiriModel::Draw();

    if (bWireFrame)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    for (UInt i = 0; i < mMeshes.size(); i++)
        mMeshes[i]->Draw();

    if (bWireFrame)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void KiriModelLoadPBR::Load(Vector3F Offset, Vector3F Scale)
{
    mOffset = Offset;
    mScale = Scale;

    Assimp::Importer importer;

    const aiScene *scene = importer.ReadFile(mPath, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        std::cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
        return;
    }

    ProcessNode(scene->mRootNode, scene);
}

void KiriModelLoadPBR::GetEdges(Array1<std::pair<Int, Int>> &Edges) const
{
    std::set<std::pair<Int, Int>> tmpCleaningBag;
    std::pair<std::set<std::pair<Int, Int>>::iterator, bool> ret1, ret2;
    for (size_t i = 0; i < mTriangles.size(); ++i)
    {
        ret1 = tmpCleaningBag.insert(std::pair<Int, Int>((Int)mTriangles[i][0], (Int)mTriangles[i][1]));
        ret2 = tmpCleaningBag.insert(std::pair<Int, Int>((Int)mTriangles[i][1], (Int)mTriangles[i][0]));
        if (ret1.second == true && ret2.second == true)
            tmpCleaningBag.erase(ret2.first);
        if (ret1.second == false && ret2.second == true)
            tmpCleaningBag.erase(ret2.first);
        if (ret1.second == true && ret2.second == false)
            tmpCleaningBag.erase(ret1.first);

        ret1 = tmpCleaningBag.insert(std::pair<Int, Int>((Int)mTriangles[i][1], (Int)mTriangles[i][2]));
        ret2 = tmpCleaningBag.insert(std::pair<Int, Int>((Int)mTriangles[i][2], (Int)mTriangles[i][1]));
        if (ret1.second == true && ret2.second == true)
            tmpCleaningBag.erase(ret2.first);
        if (ret1.second == false && ret2.second == true)
            tmpCleaningBag.erase(ret2.first);
        if (ret1.second == true && ret2.second == false)
            tmpCleaningBag.erase(ret1.first);

        ret1 = tmpCleaningBag.insert(std::pair<Int, Int>((Int)mTriangles[i][2], (Int)mTriangles[i][0]));
        ret2 = tmpCleaningBag.insert(std::pair<Int, Int>((Int)mTriangles[i][0], (Int)mTriangles[i][2]));
        if (ret1.second == true && ret2.second == true)
            tmpCleaningBag.erase(ret2.first);
        if (ret1.second == false && ret2.second == true)
            tmpCleaningBag.erase(ret2.first);
        if (ret1.second == true && ret2.second == false)
            tmpCleaningBag.erase(ret1.first);
    }
    for (std::set<std::pair<Int, Int>>::iterator it = tmpCleaningBag.begin(); it != tmpCleaningBag.end(); ++it)
        Edges.append(*it);
}

/*
   Calculate the line segment PaPb that is the shortest route between
   two lines P1P2 and P3P4. Calculate also the values of mua and mub where
      Pa = P1 + mua (P2 - P1)
      Pb = P3 + mub (P4 - P3)
   Return FALSE if no solution exists.
*/
bool KiriModelLoadPBR::LineLineIntersect(
    const Vector3F &p1, const Vector3F &p2, const Vector3F &p3, const Vector3F &p4, Vector3F &pa, Vector3F &pb,
    float &mua, float &mub)
{
    Vector3F p13, p43, p21;
    float d1343, d4321, d1321, d4343, d2121;
    float numer, denom;

    p13[0] = p1[0] - p3[0];
    p13[1] = p1[1] - p3[1];
    p13[2] = p1[2] - p3[2];
    p43[0] = p4[0] - p3[0];
    p43[1] = p4[1] - p3[1];
    p43[2] = p4[2] - p3[2];
    if (std::abs(p43[0]) < std::numeric_limits<float>::epsilon() && std::abs(p43[1]) < std::numeric_limits<float>::epsilon() && std::abs(p43[2]) < std::numeric_limits<float>::epsilon())
        return false;
    p21[0] = p2[0] - p1[0];
    p21[1] = p2[1] - p1[1];
    p21[2] = p2[2] - p1[2];
    if (std::abs(p21[0]) < std::numeric_limits<float>::epsilon() && std::abs(p21[1]) < std::numeric_limits<float>::epsilon() && std::abs(p21[2]) < std::numeric_limits<float>::epsilon())
        return false;

    d1343 = p13[0] * p43[0] + p13[1] * p43[1] + p13[2] * p43[2];
    d4321 = p43[0] * p21[0] + p43[1] * p21[1] + p43[2] * p21[2];
    d1321 = p13[0] * p21[0] + p13[1] * p21[1] + p13[2] * p21[2];
    d4343 = p43[0] * p43[0] + p43[1] * p43[1] + p43[2] * p43[2];
    d2121 = p21[0] * p21[0] + p21[1] * p21[1] + p21[2] * p21[2];

    denom = d2121 * d4343 - d4321 * d4321;
    if (std::abs(denom) < std::numeric_limits<float>::epsilon())
        return false;
    numer = d1343 * d4321 - d1321 * d4343;

    mua = numer / denom;
    mub = (d1343 + d4321 * (mua)) / d4343;

    pa[0] = p1[0] + mua * p21[0];
    pa[1] = p1[1] + mua * p21[1];
    pa[2] = p1[2] + mua * p21[2];
    pb[0] = p3[0] + mub * p43[0];
    pb[1] = p3[1] + mub * p43[1];
    pb[2] = p3[2] + mub * p43[2];

    return true;
}

bool KiriModelLoadPBR::AkinciEdgeSampling(const Vector3F &Point1, const Vector3F &Point2, const float &Radius, Array1Vec3F &Samples)
{
    Samples.clear();

    Vector3F edge = Point2 - Point1;
    float edgesL = edge.length();
    Int pNumber = std::floor(edgesL / (2.f * Radius));
    Vector3F pe = edge / (float)pNumber, p;

    for (Int j = 1; j < pNumber; ++j)
    {
        p = Point1 + (float)j * pe;
        Samples.append(p);
    }

    return true;
}

bool KiriModelLoadPBR::AkinciTriangleSampling(const Vector3F &Point1, const Vector3F &Point2, const Vector3F &Point3, const float &Radius, Array1Vec3F &Samples)
{

    std::array<Vector3F, 3> v = {{Point1, Point2, Point3}};
    std::array<Vector3F, 3> edgesV = {{v[1] - v[0], v[2] - v[1], v[0] - v[2]}};
    std::array<Vector2F, 3> edgesI = {{Vector2F(0, 1), Vector2F(1, 2), Vector2F(2, 0)}};
    std::array<float, 3> edgesL = {{edgesV[0].length(), edgesV[1].length(), edgesV[2].length()}};
    Samples.clear();

    //Triangles
    int sEdge = -1, lEdge = -1;
    float maxL = -std::numeric_limits<float>::max();
    float minL = std::numeric_limits<float>::max();
    for (int i = 0; i < 3; ++i)
    {
        if (edgesL[i] > maxL)
        {
            maxL = edgesL[i];
            lEdge = i;
        }
        if (edgesL[i] < minL)
        {
            minL = edgesL[i];
            sEdge = i;
        }
    }
    Vector3F cross, normal;
    cross = edgesV[lEdge].cross(edgesV[sEdge]);
    normal = edgesV[sEdge].cross(cross);
    normal.normalize();

    std::array<bool, 3> findVertex = {{true, true, true}};
    findVertex[(Int)edgesI[sEdge][0]] = false;
    findVertex[(Int)edgesI[sEdge][1]] = false;
    int thirdVertex = -1;
    for (size_t i = 0; i < findVertex.size(); ++i)
        if (findVertex[i] == true)
            thirdVertex = i;
    Vector3F tmpVec = v[thirdVertex] - v[(Int)edgesI[sEdge][0]];
    float sign = normal.dot(tmpVec);
    if (sign < 0)
        normal = -normal;

    float triangleHeight = std::abs(normal.dot(edgesV[lEdge]));
    int sweepSteps = triangleHeight / (2.f * Radius);
    bool success = false;

    Vector3F sweepA, sweepB, i1, i2, o1, o2;
    float m1, m2;
    int edge1, edge2;
    edge1 = (sEdge + 1) % 3;
    edge2 = (sEdge + 2) % 3;

    for (int i = 1; i < sweepSteps; ++i)
    {
        sweepA = v[(Int)edgesI[sEdge][0]] + (float)i * (2.f * Radius) * normal;
        sweepB = v[(Int)edgesI[sEdge][1]] + (float)i * (2.f * Radius) * normal;
        success = LineLineIntersect(v[(Int)edgesI[edge1][0]], v[(Int)edgesI[edge1][1]], sweepA, sweepB, o1, o2, m1, m2);
        i1 = o1;
        if (success == false)
        {
            //std::cout << "Intersection 1 failed" << std::endl;
        }
        success = LineLineIntersect(v[(Int)edgesI[edge2][0]], v[(Int)edgesI[edge2][1]], sweepA, sweepB, o1, o2, m1, m2);
        i2 = o1;
        if (success == false)
        {
            //std::cout << "Intersection 1 failed" << std::endl;
        }
        Vector3F s = i1 - i2;
        int step = std::floor(s.length() / (2.f * Radius));
        Vector3F ps = s / ((float)step);
        for (int j = 1; j < step; ++j)
        {
            Vector3F p = i2 + (float)j * ps;
            Samples.append(p);
        }
    }
    return success;
}

bool KiriModelLoadPBR::AkinciMeshSampling(const float &Radius, Array1Vec4F &Samples)
{
    bool success = true, tmp_success = false;

    //Sample Vertices
    Array1Vec3F tmp_sample;
    for (size_t i = 0; i < mVertexs.size(); ++i)
    {
        bool contains = false;
        tmp_sample.forEach([&](Vector3F elem) {
            if (elem == Vector3F(mVertexs[i].Position[0], mVertexs[i].Position[1], mVertexs[i].Position[2]))
                contains = true;
        });

        if (!contains)
        {
            tmp_sample.append(Vector3F(mVertexs[i].Position[0], mVertexs[i].Position[1], mVertexs[i].Position[2]));
            Samples.append(Vector4F(mVertexs[i].Position[0], mVertexs[i].Position[1], mVertexs[i].Position[2], Radius));
        }
    }

    //Sample edges
    tmp_sample.clear();
    Array1<std::pair<Int, Int>> edges;
    GetEdges(edges);
    for (size_t i = 0; i < edges.size(); ++i)
    {
        Vector3F P1 = Vector3F(mVertexs[edges[i].first].Position[0], mVertexs[edges[i].first].Position[1], mVertexs[edges[i].first].Position[2]);
        Vector3F P2 = Vector3F(mVertexs[edges[i].second].Position[0], mVertexs[edges[i].second].Position[1], mVertexs[edges[i].second].Position[2]);
        tmp_success = AkinciEdgeSampling(P1, P2, Radius, tmp_sample);
        for (size_t j = 0; j < tmp_sample.size(); ++j)
        {
            Vector4F p(tmp_sample[j].x, tmp_sample[j].y, tmp_sample[j].z, Radius);
            Samples.append(p);
        }

        success = success && tmp_success;
    }

    //Sample triangles
    tmp_sample.clear();
    for (size_t i = 0; i < mTriangles.size(); ++i)
    {
        Vector3F P1 = Vector3F(mVertexs[mTriangles[i][0]].Position[0], mVertexs[mTriangles[i][0]].Position[1], mVertexs[mTriangles[i][0]].Position[2]);
        Vector3F P2 = Vector3F(mVertexs[mTriangles[i][1]].Position[0], mVertexs[mTriangles[i][1]].Position[1], mVertexs[mTriangles[i][1]].Position[2]);
        Vector3F P3 = Vector3F(mVertexs[mTriangles[i][2]].Position[0], mVertexs[mTriangles[i][2]].Position[1], mVertexs[mTriangles[i][2]].Position[2]);

        tmp_success = AkinciTriangleSampling(P1, P2, P3, Radius, tmp_sample);
        for (size_t j = 0; j < tmp_sample.size(); ++j)
            Samples.append(Vector4F(tmp_sample[j].x, tmp_sample[j].y, tmp_sample[j].z, Radius));
        success = success && tmp_success;
    }

    KIRI_LOG_INFO("Vertex Number={0:d}", mVertexs.size());
    KIRI_LOG_INFO("Sampling Points Number={0:d}", Samples.size());
    return success;
}

void KiriModelLoadPBR::ProcessNode(aiNode *node, const aiScene *scene)
{
    // process each mesh located at the current node
    for (UInt i = 0; i < node->mNumMeshes; i++)
    {
        // the node object only contains indices to index the actual objects in the scene.
        // the scene contains all the data, node is just to keep stuff organized (like relations between nodes).
        aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
        mMeshes.append(ProcessMesh(mesh, scene));
    }
    // after we've processed all of the mMeshes (if any) we then recursively process each of the children nodes
    for (UInt i = 0; i < node->mNumChildren; i++)
    {
        ProcessNode(node->mChildren[i], scene);
    }
}

void KiriModelLoadPBR::ReMakeVertexPosition(Vector3F Offset, Vector3F Scale, bool Normalize)
{
    //TODO need check aspect
    auto diff = mAABBMax - mAABBMin;
    float maxSize = fmaxf(fmaxf(std::abs(diff[0]), std::abs(diff[1])), std::abs(diff[2]));
    Vector3F scale = Vector3F(1.f);
    if (Normalize)
        scale = Scale / maxSize;

    mAABBMin *= scale;
    mAABBMax *= scale;

    // len(max(x,y,z)) = 1 and move mesh center to origin
    auto meshCenter = (mAABBMax + mAABBMin) * 0.5f - Offset;

    mAABBMin = Vector3F(1e10f);
    mAABBMax = Vector3F(-1e10f);
    for (size_t i = 0, iend = mVertexs.size(); i < iend; ++i)
    {
        for (size_t k = 0; k < 3; k++)
        {
            mVertexs[i].Position[k] = mVertexs[i].Position[k] * Scale[k];
            mVertexs[i].Position[k] = mVertexs[i].Position[k] - meshCenter[k];

            mAABBMin[k] = fminf(mAABBMin[k], mVertexs[i].Position[k]);
            mAABBMax[k] = fmaxf(mAABBMax[k], mVertexs[i].Position[k]);
        }
    }

    float expandRatio = 0.05f;
    meshCenter = (mAABBMax + mAABBMin) * 0.5f;
    Vector3F expandHalfLength = (mAABBMax - mAABBMin) * 0.5f * (1.f + expandRatio);
    mBBoxMin = meshCenter - expandHalfLength;
    mBBoxMax = meshCenter + expandHalfLength;
}

KiriMeshPBRPtr KiriModelLoadPBR::ProcessMesh(aiMesh *mesh, const aiScene *scene)
{
    // vertex
    for (UInt i = 0; i < mesh->mNumVertices; i++)
    {
        VertexFull vertex;

        // positions
        vertex.Position[0] = mesh->mVertices[i].x;
        vertex.Position[1] = mesh->mVertices[i].y;
        vertex.Position[2] = mesh->mVertices[i].z;

        // find AABB
        mAABBMin[0] = std::min(vertex.Position[0], mAABBMin[0]);
        mAABBMin[1] = std::min(vertex.Position[1], mAABBMin[1]);
        mAABBMin[2] = std::min(vertex.Position[2], mAABBMin[2]);

        mAABBMax[0] = std::max(vertex.Position[0], mAABBMax[0]);
        mAABBMax[1] = std::max(vertex.Position[1], mAABBMax[1]);
        mAABBMax[2] = std::max(vertex.Position[2], mAABBMax[2]);

        // normals
        vertex.Normal[0] = mesh->mNormals[i].x;
        vertex.Normal[1] = mesh->mNormals[i].y;
        vertex.Normal[2] = mesh->mNormals[i].z;

        // texture coordinates
        if (mesh->mTextureCoords[0])
        {

            vertex.TexCoords[0] = mesh->mTextureCoords[0][i].x;
            vertex.TexCoords[1] = mesh->mTextureCoords[0][i].y;
        }
        else
        {
            vertex.TexCoords[0] = 0.0f;
            vertex.TexCoords[1] = 0.0f;
        }
        // tangent
        vertex.Tangent[0] = mesh->mTangents[i].x;
        vertex.Tangent[1] = mesh->mTangents[i].y;
        vertex.Tangent[2] = mesh->mTangents[i].z;

        // bitangent
        vertex.Bitangent[0] = mesh->mBitangents[i].x;
        vertex.Bitangent[1] = mesh->mBitangents[i].y;
        vertex.Bitangent[2] = mesh->mBitangents[i].z;

        mVertexs.append(vertex);
    }

    // face
    for (UInt i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];
        // retrieve all indices of the face and store them in the indices vector
        for (UInt j = 0; j < face.mNumIndices; j++)
            mIndices.append(face.mIndices[j]);

        mTriangles.append(Vector3F(face.mIndices[0], face.mIndices[1], face.mIndices[2]));
    }

    ReMakeVertexPosition(mOffset, mScale);

    return std::make_shared<KiriMeshPBR>(mVertexs, mIndices);
}