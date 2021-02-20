/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-21 17:15:03
 * @LastEditTime: 2020-11-04 23:25:26
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\geo\geo_object.cpp
 */

#include <kiri_core/geo/geo_object.h>
#include <kiri_core/geo/geo_helper.h>
#include <kiri_core/model/model_tiny_obj_loader.h>

Vector3F KiriGeoObject::transform(const Vector3F &ppos) const
{
    if (!mTransformed)
    {
        return ppos;
    }
    else
    {
        //TODO
        //return VecX<N, RealType>(m_TransformationMatrix * VecX<N + 1, RealType>(ppos, 1.0));
        return Vector3F(0.f);
    }
}

KiriTriMeshObject::KiriTriMeshObject(const String &meshFilePath, float sdfStep, Vector3F Offset, float BoxScale)
{
    mTriMeshFile = meshFilePath;
    mStep = sdfStep;
    mOffset = Offset;
    mBoxScale = BoxScale;
    computeSDF();
}

void KiriTriMeshObject::computeSDF()
{
    KiriModelTinyObjLoaderPtr meshLoader = std::make_shared<KiriModelTinyObjLoader>(mTriMeshFile, "models", ".obj");
    meshLoader->scaleToBox(mBoxScale);

    mAABBMin = meshLoader->getAABBMin();
    mAABBMax = meshLoader->getAABBMax();

    mGrid3D.SetGrid(meshLoader->getAABBMin() - Vector3F(3.f * mStep),
                    meshLoader->getAABBMax() + Vector3F(3.f * mStep),
                    mStep);

    Vec_Vec3F vertexList(meshLoader->getNVertices());
    Vec_Vec3F faceList(meshLoader->getNFaces());

    std::memcpy(vertexList.data(), meshLoader->getVertices().data(), meshLoader->getVertices().size() * sizeof(float));
    std::memcpy(faceList.data(), meshLoader->getFaces().data(), meshLoader->getFaces().size() * sizeof(float));

    // KIRI_LOG_DEBUG("AABBMin=({0},{1},{2}),AABBMAX=({3},{4},{5})",
    //                meshLoader->getAABBMin().x, meshLoader->getAABBMin().y, meshLoader->getAABBMin().z,
    //                meshLoader->getAABBMax().x, meshLoader->getAABBMax().y, meshLoader->getAABBMax().z);

    // Compute SDF data
    computeSDFMesh(faceList, vertexList,
                   meshLoader->getAABBMin(), mStep, mGrid3D.getNCells()[0], mGrid3D.getNCells()[1], mGrid3D.getNCells()[2], mSDFData);
    mSDFGenerated = true;
}

// sign distance field for triangle mesh
void KiriTriMeshObject::computeSDFMesh(const Vec_Vec3F &faces, const Vec_Vec3F &vertices, const Vector3F &origin, float CellSize,
                                       float ni, float nj, float nk, Array3F &SDF, Int exactBand)
{
    //KIRI_LOG_INFO("KiriMath Max Thread={0:d}", kiri_math::maxNumberOfThreads());
    kiri_math::setMaxNumberOfThreads(kiri_math::maxNumberOfThreads());
    //check ni nj nk
    SDF.resize(ni, nj, nk, (ni + nj + nk) * CellSize); // upper bound on distance

    Array3UI closest_tri(ni, nj, nk, 0xffffffff);

    // intersection_count(i,j,k) is # of tri intersections in (i-1,i]x{j}x{k}
    // we begin by initializing distances near the mesh, and figuring out intersection counts
    Array3UI intersectionCount(ni, nj, nk, 0u);

    for (UInt face = 0, faceEnd = static_cast<UInt>(faces.size()); face < faceEnd; ++face)
    {
        UInt p = faces[face][0];
        UInt q = faces[face][1];
        UInt r = faces[face][2];

        // coordinates in grid to high precision
        Vector3F fp = (vertices[p] - origin) / CellSize;
        Vector3F fq = (vertices[q] - origin) / CellSize;
        Vector3F fr = (vertices[r] - origin) / CellSize;

        // do distances nearby
        Int i0 = kiri_math::clamp(static_cast<Int>(kiri_math::min3(fp[0], fq[0], fr[0])) - exactBand, 0, static_cast<Int>(ni - 1));
        Int i1 = kiri_math::clamp(static_cast<Int>(kiri_math::max3(fp[0], fq[0], fr[0])) + exactBand + 1, 0, static_cast<Int>(ni - 1));
        Int j0 = kiri_math::clamp(static_cast<Int>(kiri_math::min3(fp[1], fq[1], fr[1])) - exactBand, 0, static_cast<Int>(nj - 1));
        Int j1 = kiri_math::clamp(static_cast<Int>(kiri_math::max3(fp[1], fq[1], fr[1])) + exactBand + 1, 0, static_cast<Int>(nj - 1));
        Int k0 = kiri_math::clamp(static_cast<Int>(kiri_math::min3(fp[2], fq[2], fr[2])) - exactBand, 0, static_cast<Int>(nk - 1));
        Int k1 = kiri_math::clamp(static_cast<Int>(kiri_math::max3(fp[2], fq[2], fr[2])) + exactBand + 1, 0, static_cast<Int>(nk - 1));

        kiri_math::parallelFor(i0, i1 + 1, j0, j1 + 1, k0, k1 + 1, [&](Int i, Int j, Int k) {
            Vector3F gx = Vector3F(i, j, k) * CellSize + origin;
            float d = KIRI::point_triangle_distance(gx, vertices[p], vertices[q], vertices[r]);

            if (d < SDF(i, j, k))
            {
                SDF(i, j, k) = d;
                closest_tri(i, j, k) = face;
            }
        });

        Int expand_val = 1;
        // and do intersection counts
        j0 = kiri_math::clamp(static_cast<Int>(std::ceil(kiri_math::min3(fp[1], fq[1], fr[1]))) - expand_val, 0, static_cast<Int>(nj - 1));
        j1 = kiri_math::clamp(static_cast<Int>(std::floor(kiri_math::max3(fp[1], fq[1], fr[1]))) + expand_val, 0, static_cast<Int>(nj - 1));
        k0 = kiri_math::clamp(static_cast<Int>(std::ceil(kiri_math::min3(fp[2], fq[2], fr[2]))) - expand_val, 0, static_cast<Int>(nk - 1));
        k1 = kiri_math::clamp(static_cast<Int>(std::floor(kiri_math::max3(fp[2], fq[2], fr[2]))) + expand_val, 0, static_cast<Int>(nk - 1));

        for (Int k = k0; k <= k1; ++k)
        {

            for (Int j = j0; j <= j1; ++j)
            {
                float a, b, c;

                if (KIRI::point_in_triangle_2d(static_cast<float>(j), static_cast<float>(k), fp[1], fp[2], fq[1], fq[2], fr[1], fr[2], a, b, c))
                {
                    // intersection i coordinate
                    float fi = a * fp[0] + b * fq[0] + c * fr[0];

                    // intersection is in (i_interval-1,i_interval]
                    Int i_interval = std::max(static_cast<Int>(std::ceil(fi)), 0);

                    // we enlarge the first interval to include everything to the -x direction
                    // we ignore intersections that are beyond the +x side of the grid
                    if (i_interval < static_cast<Int>(ni))
                    {
                        ++intersectionCount(i_interval, j, k);
                    }
                }
            }
        }
    } // end loop face

    // and now we fill in the rest of the distances with fast sweeping
    for (UInt pass = 0; pass < 2; ++pass)
    {
        KIRI::sweep(faces, vertices, SDF, closest_tri, origin, CellSize, +1, +1, +1);
        KIRI::sweep(faces, vertices, SDF, closest_tri, origin, CellSize, -1, -1, -1);
        KIRI::sweep(faces, vertices, SDF, closest_tri, origin, CellSize, +1, +1, -1);
        KIRI::sweep(faces, vertices, SDF, closest_tri, origin, CellSize, -1, -1, +1);
        KIRI::sweep(faces, vertices, SDF, closest_tri, origin, CellSize, +1, -1, +1);
        KIRI::sweep(faces, vertices, SDF, closest_tri, origin, CellSize, -1, +1, -1);
        KIRI::sweep(faces, vertices, SDF, closest_tri, origin, CellSize, +1, -1, -1);
        KIRI::sweep(faces, vertices, SDF, closest_tri, origin, CellSize, -1, +1, +1);
    }

    kiri_math::parallelFor<UInt>(0, static_cast<UInt>(nk), [&](UInt k) {
        for (UInt j = 0; j < nj; ++j)
        {
            UInt total_count = 0;

            for (UInt i = 0; i < ni; ++i)
            {
                total_count += intersectionCount(i, j, k);

                // intersection count odd
                if (total_count & 1)
                {
                    //KIRI_LOG_DEBUG("total_count={0}", total_count);
                    SDF(i, j, k) = -SDF(i, j, k); // we are inside the mesh
                }
            }
        }
    });
}

float KiriTriMeshObject::signedDistance(const Vector3F &ppos0, bool bNegativeInside) const
{

    KIRI_ASSERT(mSDFGenerated);

    //TODO transform
    //auto ppos = this->invTransform(ppos0);
    auto ppos = ppos0;

    auto gridPos = mGrid3D.getGridCoordinate(ppos);
    float d = this->mUniformScale * KIRI::interpolateValueLinear(gridPos, mSDFData);
    return bNegativeInside ? d : -d;
}
