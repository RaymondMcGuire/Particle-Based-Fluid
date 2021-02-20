/*** 
 * @Author: Pierre-Luc Manteaux
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2020-12-30 02:43:27
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\sampler\kiri_sampler_basic.cpp
 * @Reference: https://github.com/manteapi/hokusai
 */

#include <kiri_core/sampler/kiri_sampler_basic.h>

KiriSamplerBasic::KiriSamplerBasic()
{
}

Array1Vec3F KiriSamplerBasic::GetCubeSampling(Vector3F center, Vector3F sides, float spacing)
{
    mPoints.clear();

    for (Int i = 0; i < sides.x; ++i)
    {
        for (Int j = 0; j < sides.y; ++j)
        {
            for (Int k = 0; k < sides.z; ++k)
            {
                mPoints.append(center + Vector3F(i, j, k) * spacing);
            }
        }
    }

    return mPoints;
}

Array1Vec3F KiriSamplerBasic::GetBoxSampling(Vector3F center, Vector3F sides, float spacing)
{
    mPoints.clear();

    Int epsilon = 0;
    Vector3F lengths = sides * spacing;

    //ZX plane - bottom
    for (Int i = -epsilon; i <= sides.x + epsilon; ++i)
    {
        for (Int j = -epsilon; j <= sides.z + epsilon; ++j)
        {
            mPoints.append(Vector3F(center.x + i * spacing, center.y, center.z + j * spacing));
        }
    }

    //ZX plane - top
    for (Int i = -epsilon; i <= sides.x + epsilon; ++i)
    {
        for (Int j = -epsilon; j <= sides.z + epsilon; ++j)
        {
            mPoints.append(Vector3F(center.x + i * spacing, center.y + lengths.y, center.z + j * spacing));
        }
    }

    //XY plane - back
    for (Int i = -epsilon; i <= sides.x + epsilon; ++i)
    {
        for (Int j = -epsilon; j <= sides.y + epsilon; ++j)
        {
            mPoints.append(Vector3F(center.x + i * spacing, center.y + j * spacing, center.z));
        }
    }

    //XY plane - front
    for (Int i = -epsilon; i <= sides.x + epsilon; ++i)
    {
        for (Int j = -epsilon; j <= sides.y - epsilon; ++j)
        {
            mPoints.append(Vector3F(center.x + i * spacing, center.y + j * spacing, center.z + lengths.z));
        }
    }

    //YZ plane - left
    for (Int i = -epsilon; i <= sides.y + epsilon; ++i)
    {
        for (Int j = -epsilon; j <= sides.z + epsilon; ++j)
        {
            mPoints.append(Vector3F(center.x, center.y + i * spacing, center.z + j * spacing));
        }
    }

    //YZ plane - right
    for (Int i = -epsilon; i <= sides.y + epsilon; ++i)
    {
        for (Int j = -epsilon; j <= sides.z + epsilon; ++j)
        {
            mPoints.append(Vector3F(center.x + lengths.x, center.y + i * spacing, center.z + j * spacing));
        }
    }
    return mPoints;
}

Array1Vec3F KiriSamplerBasic::GetDiskSampling(Vector3F center, float radius, float spacing)
{
    mPoints.clear();

    float theta = 0.0;
    float u = 0.0;
    float l_theta = 2.0 * kiri_math::pi<float>() * radius;
    Int thetaStep = std::floor(l_theta / spacing);
    Int uStep = std::floor(radius / spacing);

    for (Int i = 0; i < thetaStep; ++i)
    {
        theta += (2.0 * kiri_math::pi<float>() / (float)thetaStep);
        u = 0.0;
        for (Int j = 0; j < uStep; ++j)
        {
            u += (radius / (float)uStep);
            mPoints.append(center + Vector3F(u * radius * cos(theta), u * radius * sin(theta), 0));
        }
    }
    return mPoints;
}

Array1Vec3F KiriSamplerBasic::GetSphereSampling(const Vector3F center, float radius, float spacingX, float spacingY)
{
    mPoints.clear();

    float theta = 0.0;
    float phi = 0.0;
    float l_theta = 2.0 * kiri_math::pi<float>() * radius;
    float l_phi = kiri_math::pi<float>() * radius;
    Int thetaStep = std::floor(l_theta / spacingX);
    Int phiStep = std::floor(l_phi / spacingY);
    Array1Vec3F result;

    for (Int i = 0; i < thetaStep; ++i)
    {
        theta += (2.0 * kiri_math::pi<float>() / (float)thetaStep);
        phi = 0.0;
        for (Int j = 0; j < phiStep; ++j)
        {
            phi += (kiri_math::pi<float>() / (float)phiStep);
            mPoints.append(center + Vector3F(radius * cos(theta) * sin(phi), radius * sin(theta) * sin(phi), radius * cos(phi)));
        }
    }
    return mPoints;
}

Array1Vec3F KiriSamplerBasic::GetEllipsoidSampling(const Vector3F center, float axis_1, float axis_2, float axis_3, float spacingX, float spacingY)
{
    float theta = 0.0;
    float phi = 0.0;
    float l_theta = 2.0 * kiri_math::pi<float>() * axis_1;
    float l_phi = kiri_math::pi<float>() * axis_2;
    Int thetaStep = std::floor(l_theta / spacingX);
    Int phiStep = std::floor(l_phi / spacingY);
    Array1Vec3F result;

    for (Int i = 0; i < thetaStep; ++i)
    {
        theta += (2.0 * kiri_math::pi<float>() / (float)thetaStep);
        phi = 0.0;
        for (Int j = 0; j < phiStep; ++j)
        {
            phi += (kiri_math::pi<float>() / (float)phiStep);
            mPoints.append(center + Vector3F(axis_1 * cos(theta) * sin(phi), axis_2 * sin(theta) * sin(phi), axis_3 * cos(phi)));
        }
    }
    return mPoints;
}

Array1Vec3F KiriSamplerBasic::GetCapsuleSampling(const Vector3F center, float radius, float height, float spacingX, float spacingY)
{
    float theta = 0.0;
    float phi = 0.0;
    float l_theta = 2.0 * kiri_math::pi<float>() * radius;
    float l_phi = (kiri_math::pi<float>() / 2.0) * radius;
    Int thetaStep = std::floor(l_theta / spacingX);
    Int phiStep = std::floor(l_phi / spacingY);
    Array1Vec3F result;
    Vector3F c1(center[0], center[2], height);
    Vector3F c2(center[0], center[2], 0);

    for (Int i = 0; i < thetaStep; ++i)
    {
        theta += (2.0 * kiri_math::pi<float>() / (float)thetaStep);
        phi = 0.0;
        for (Int j = 0; j < phiStep; ++j)
        {
            phi += (kiri_math::pi<float>() / (float)(2.0 * phiStep));
            mPoints.append(c1 + Vector3F(radius * cos(theta) * sin(phi), radius * sin(theta) * sin(phi), radius * cos(phi)));
        }
    }

    for (Int i = 0; i < thetaStep; ++i)
    {
        theta += (2.0 * kiri_math::pi<float>() / (float)thetaStep);
        phi = kiri_math::pi<float>() / 2.0 - (kiri_math::pi<float>() / (2.0 * phiStep));
        for (Int j = 0; j < phiStep; ++j)
        {
            phi += (kiri_math::pi<float>() / (float)(2.0 * (phiStep)));
            mPoints.append(c2 + Vector3F(radius * cos(theta) * sin(phi), radius * sin(theta) * sin(phi), radius * cos(phi)));
        }
    }

    Array1Vec3F tmp = GetCylinderSampling(center, height, radius, spacingX, spacingY);
    for (size_t i = 0; i < tmp.size(); ++i)
        mPoints.append(tmp[i]);

    return mPoints;
}

Array1Vec3F KiriSamplerBasic::GetHemiSphereSampling(const Vector3F center, float radius, float spacingX, float spacingY)
{
    float theta = 0.0;
    float phi = 0.0;
    float l_theta = 2.0 * kiri_math::pi<float>() * radius;
    float l_phi = (kiri_math::pi<float>() / 2.0) * radius;
    Int thetaStep = std::floor(l_theta / spacingX);
    Int phiStep = std::floor(l_phi / spacingY);
    Array1Vec3F result;

    for (Int i = 0; i < thetaStep; ++i)
    {
        theta += (2.0 * kiri_math::pi<float>() / (float)thetaStep);
        phi = 0.0;
        for (Int j = 0; j < phiStep; ++j)
        {
            phi += (kiri_math::pi<float>() / (float)(2.0 * phiStep));
            mPoints.append(center + Vector3F(radius * cos(theta) * sin(phi), radius * sin(theta) * sin(phi), radius * cos(phi)));
        }
    }
    return mPoints;
}

Array1Vec3F KiriSamplerBasic::GetTorusSampling(const Vector3F center, float tubeRadius, float innerRadius, float spacingX, float spacingY)
{
    float u = 0.0;
    float v = 0.0;
    float l_u = 2.0 * kiri_math::pi<float>() * innerRadius;
    float l_v = 2.0 * kiri_math::pi<float>() * tubeRadius;
    Int uStep = std::floor(l_u / spacingX);
    Int vStep = std::floor(l_v / spacingY);
    Array1Vec3F result;

    for (Int i = 0; i < uStep; ++i)
    {
        u += (2.0 * kiri_math::pi<float>() / (float)uStep);
        v = 0.0;
        for (Int j = 0; j < vStep; ++j)
        {
            v += (2.0 * kiri_math::pi<float>() / (float)vStep);
            mPoints.append(center + Vector3F((innerRadius + tubeRadius * cos(v)) * cos(u), (innerRadius + tubeRadius * cos(v)) * sin(u), tubeRadius * sin(v)));
        }
    }
    return mPoints;
}

Array1Vec3F KiriSamplerBasic::GetConeSampling(const Vector3F center, float height, float stopHeight, float baseRadius, float spacingX, float spacingY)
{
    float theta = 0.0;
    float u = 0.0;
    float l_theta = 2.0 * kiri_math::pi<float>() * baseRadius;
    Int thetaStep = std::floor(l_theta / spacingX);
    Int uStep = std::floor(stopHeight / spacingY);
    Array1Vec3F result;

    for (Int i = 0; i < thetaStep; ++i)
    {
        theta += (2.0 * kiri_math::pi<float>() / (float)thetaStep);
        u = 0.0;
        for (Int j = 0; j < uStep; ++j)
        {
            u += (stopHeight / (float)uStep);
            mPoints.append(center + Vector3F(((height - u) / height) * baseRadius * cos(theta), ((height - u) / height) * baseRadius * sin(theta), u));
        }
    }
    return mPoints;
}

Array1Vec3F KiriSamplerBasic::GetCylinderSampling(const Vector3F center, float height, float baseRadius, float spacingX, float spacingY)
{
    float theta = 0.0;
    float u = 0.0;
    float l_theta = 2.0 * kiri_math::pi<float>() * baseRadius;
    Int thetaStep = std::floor(l_theta / spacingX);
    Int uStep = std::floor(height / spacingY);
    Array1Vec3F result;

    for (Int i = 0; i < thetaStep; ++i)
    {
        theta += (2.0 * kiri_math::pi<float>() / (float)thetaStep);
        u = 0.0;
        for (Int j = 0; j < uStep; ++j)
        {
            u += (height / (float)uStep);
            mPoints.append(center + Vector3F(baseRadius * cos(theta), baseRadius * sin(theta), u));
        }
    }
    return mPoints;
}

Array1Vec4F KiriSamplerBasic::GetCubeSamplingWithRadius(Vector3F center, Vector3F sides, float spacing)
{
    Array1Vec4F pointsWithRadius;
    Int cnt = 1;
    Int maxNum = sides.x * sides.y * sides.z;
    for (Int i = 0; i < sides.x; ++i)
    {
        for (Int j = 0; j < sides.y; ++j)
        {
            for (Int k = 0; k < sides.z; ++k)
            {
                float radius = spacing / 2.f;
                Vector3F pos = center + Vector3F(i, j, k) * spacing;
                pointsWithRadius.append(Vector4F(pos.x, pos.y, pos.z, radius * cnt / maxNum));
                cnt++;
            }
        }
    }

    return pointsWithRadius;
}

/*
   Calculate the line segment PaPb that is the shortest route between
   two lines P1P2 and P3P4. Calculate also the values of mua and mub where
      Pa = P1 + mua (P2 - P1)
      Pb = P3 + mub (P4 - P3)
   Return FALSE if no solution exists.
*/
bool KiriSamplerBasic::LineLineIntersect(
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

bool KiriSamplerBasic::AkinciEdgeSampling(const Vector3F &Point1, const Vector3F &Point2, const float &Radius, Array1Vec3F &Samples)
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

bool KiriSamplerBasic::AkinciTriangleSampling(const Vector3F &Point1, const Vector3F &Point2, const Vector3F &Point3, const float &Radius, Array1Vec3F &Samples)
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
            // //std::cout << "Intersection 1 failed" << std::endl;
        }
        success = LineLineIntersect(v[(Int)edgesI[edge2][0]], v[(Int)edgesI[edge2][1]], sweepA, sweepB, o1, o2, m1, m2);
        i2 = o1;
        if (success == false)
        {
            // //std::cout << "Intersection 1 failed" << std::endl;
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

bool KiriSamplerBasic::AkinciMeshSampling(const KiriMeshTriangle *Mesh, const float &Radius, Array1Vec4F &Samples)
{
    bool success = true, tmp_success = false;

    //Sample Vertices
    Array1Vec3F tmp_sample;
    for (size_t i = 0; i < Mesh->vertices().size(); ++i)
    {
        bool contains = false;
        tmp_sample.forEach([&](Vector3F elem) {
            if (elem == Vector3F(Mesh->vertices()[i].x, Mesh->vertices()[i].y, Mesh->vertices()[i].z))
                contains = true;
        });

        if (!contains)
        {
            tmp_sample.append(Vector3F(Mesh->vertices()[i].x, Mesh->vertices()[i].y, Mesh->vertices()[i].z));
            Samples.append(Vector4F(Mesh->vertices()[i].x, Mesh->vertices()[i].y, Mesh->vertices()[i].z, Radius));
        }
    }

    //Sample edges
    tmp_sample.clear();
    Array1<std::pair<Int, Int>> edges;
    Mesh->GetEdges(edges);
    for (size_t i = 0; i < edges.size(); ++i)
    {
        tmp_success = AkinciEdgeSampling(Mesh->vertices()[edges[i].first], Mesh->vertices()[edges[i].second], Radius, tmp_sample);
        for (size_t j = 0; j < tmp_sample.size(); ++j)
        {
            Vector4F p(tmp_sample[j].x, tmp_sample[j].y, tmp_sample[j].z, Radius);
            Samples.append(p);
        }

        success = success && tmp_success;
    }

    //Sample triangles
    tmp_sample.clear();
    for (size_t i = 0; i < Mesh->triangles().size(); ++i)
    {
        tmp_success = AkinciTriangleSampling(Mesh->vertices()[Mesh->triangles()[i][0]], Mesh->vertices()[Mesh->triangles()[i][1]], Mesh->vertices()[Mesh->triangles()[i][2]], Radius, tmp_sample);
        for (size_t j = 0; j < tmp_sample.size(); ++j)
            Samples.append(Vector4F(tmp_sample[j].x, tmp_sample[j].y, tmp_sample[j].z, Radius));
        success = success && tmp_success;
    }

    KIRI_LOG_INFO("Vertex Number={0:d}", Mesh->vertices().size());
    KIRI_LOG_INFO("Sampling Points Number={0:d}", Samples.size());
    return success;
}