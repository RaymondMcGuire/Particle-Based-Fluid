/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:52:45
 * @FilePath: \core\src\kiri_core\mesh\mesh_cylinder.cpp
 * @Description: reference http://www.songho.ca/opengl/gl_cylinder.html
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_core/mesh/mesh_cylinder.h>

void KiriMeshCylinder::BuildUnitCircleVertices()
{
    float sectorStep = 2 * kiri_math::pi<float>() / mSectorCount;
    float sectorAngle; // radian

    mUnitCircleVertices.clear();
    for (Int i = 0; i <= mSectorCount; ++i)
    {
        sectorAngle = i * sectorStep;
        mUnitCircleVertices.append(cos(sectorAngle)); // x
        mUnitCircleVertices.append(sin(sectorAngle)); // y
        mUnitCircleVertices.append(0);                // z
    }
}

Array1<float> KiriMeshCylinder::GetSideNormals()
{
    float sectorStep = 2 * kiri_math::pi<float>() / mSectorCount;
    float sectorAngle; // radian

    // compute the mNormal vector at 0 degree first
    // tanA = (mBaseRadius-mTopRadius) / height
    float zAngle = atan2(mBaseRadius - mTopRadius, mHeight);
    float x0 = cos(zAngle); // nx
    float y0 = 0;           // ny
    float z0 = sin(zAngle); // nz

    // Rotate (x0,y0,z0) per sector angle
    Array1<float> normals;
    for (Int i = 0; i <= mSectorCount; ++i)
    {
        sectorAngle = i * sectorStep;
        normals.append(cos(sectorAngle) * x0 - sin(sectorAngle) * y0); // nx
        normals.append(sin(sectorAngle) * x0 + cos(sectorAngle) * y0); // ny
        normals.append(z0);                                            // nz
    }

    return normals;
}

void KiriMeshCylinder::BuildVerticesSmooth()
{
    float x, y, z; // vertex position
    // float s, t;                                     // texCoord
    float radius; // radius for each stack

    // get normals for cylinder sides
    Array1<float> sideNormals = GetSideNormals();

    // put vertices of mSide cylinder to array by scaling unit circle
    for (Int i = 0; i <= mStackCount; ++i)
    {
        z = -(mHeight * 0.5f) + (float)i / mStackCount * mHeight;                   // vertex position z
        radius = mBaseRadius + (float)i / mStackCount * (mTopRadius - mBaseRadius); // lerp
        float t = 1.0f - (float)i / mStackCount;                                    // top-to-bottom

        for (Int j = 0, k = 0; j <= mSectorCount; ++j, k += 3)
        {
            x = mUnitCircleVertices[k];
            y = mUnitCircleVertices[k + 1];

            AddVertStand(Vector3F(x * radius, y * radius, z), Vector3F(sideNormals[k], sideNormals[k + 1], sideNormals[k + 2]), Vector2F((float)j / mSectorCount, t));
        }
    }

    // remember where the base.top vertices start
    UInt baseVertexIndex = (UInt)mVertStand.size();

    // put vertices of base of cylinder
    z = -mHeight * 0.5f;
    AddVertStand(Vector3F(0, 0, z), Vector3F(0, 0, -1), Vector2F(0.5f, 0.5f));
    for (Int i = 0, j = 0; i < mSectorCount; ++i, j += 3)
    {
        x = mUnitCircleVertices[j];
        y = mUnitCircleVertices[j + 1];
        // texcoord flip horizontal
        AddVertStand(Vector3F(x * mBaseRadius, y * mBaseRadius, z), Vector3F(0, 0, -1), Vector2F(-x * 0.5f + 0.5f, -y * 0.5f + 0.5f));
    }

    // remember where the base vertices start
    UInt topVertexIndex = (UInt)mVertStand.size();

    // put vertices of top of cylinder
    z = mHeight * 0.5f;
    AddVertStand(Vector3F(0, 0, z), Vector3F(0, 0, 1), Vector2F(0.5f, 0.5f));
    for (Int i = 0, j = 0; i < mSectorCount; ++i, j += 3)
    {
        x = mUnitCircleVertices[j];
        y = mUnitCircleVertices[j + 1];
        AddVertStand(Vector3F(x * mTopRadius, y * mTopRadius, z), Vector3F(0, 0, 1), Vector2F(x * 0.5f + 0.5f, y * 0.5f + 0.5f));
    }

    // put mIndices for sides
    UInt k1, k2;
    for (Int i = 0; i < mStackCount; ++i)
    {
        k1 = i * (mSectorCount + 1); // bebinning of current stack
        k2 = k1 + mSectorCount + 1;  // beginning of next stack

        for (Int j = 0; j < mSectorCount; ++j, ++k1, ++k2)
        {
            // 2 trianles per sector
            mIndices.append(k1);
            mIndices.append(k1 + 1);
            mIndices.append(k2);

            mIndices.append(k2);
            mIndices.append(k1 + 1);
            mIndices.append(k2 + 1);

            // vertical lines for all stacks
            lineIndices.append(k1);
            lineIndices.append(k2);
            // horizontal lines
            lineIndices.append(k2);
            lineIndices.append(k2 + 1);
            if (i == 0)
            {
                lineIndices.append(k1);
                lineIndices.append(k1 + 1);
            }
        }
    }

    // remember where the base mIndices start
    mBaseIndex = (UInt)mIndices.size();

    // put mIndices for base
    for (Int i = 0, k = baseVertexIndex + 1; i < mSectorCount; ++i, ++k)
    {
        if (i < (mSectorCount - 1))
        {
            mIndices.append(baseVertexIndex);
            mIndices.append(k + 1);
            mIndices.append(k);
        }
        else
        {
            mIndices.append(baseVertexIndex);
            mIndices.append(baseVertexIndex + 1);
            mIndices.append(k);
        }
    }

    // remember where the base mIndices start
    mTopIndex = (UInt)mIndices.size();

    for (Int i = 0, k = topVertexIndex + 1; i < mSectorCount; ++i, ++k)
    {
        if (i < (mSectorCount - 1))
        {
            mIndices.append(topVertexIndex);
            mIndices.append(k);
            mIndices.append(k + 1);
        }
        else
        {
            mIndices.append(topVertexIndex);
            mIndices.append(k);
            mIndices.append(topVertexIndex + 1);
        }
    }
}

void KiriMeshCylinder::Construct()
{
    mDrawElem = true;
    mVertDataType = DataType::Standard;

    BuildUnitCircleVertices();

    if (mSmooth)
    {
        BuildVerticesSmooth();
    }

    mVerticesNum = mVertStand.size();

    SetupVertex();
}

KiriMeshCylinder::KiriMeshCylinder(float baseRadius, float topRadius, float height, Int sectors,
                                   Int stacks, bool smooth)
{
    mInstance = false;

    this->mBaseRadius = baseRadius;
    this->mTopRadius = topRadius;
    this->mHeight = height;
    this->mSectorCount = sectors;
    if (sectors < MIN_SECTOR_COUNT)
        this->mSectorCount = MIN_SECTOR_COUNT;
    this->mStackCount = stacks;
    if (stacks < MIN_STACK_COUNT)
        this->mStackCount = MIN_STACK_COUNT;
    this->mSmooth = smooth;

    Construct();
}

void KiriMeshCylinder::Draw()
{

    glBindVertexArray(mVAO);
    if (!mInstance)
    {
        glDrawElements(GL_TRIANGLES, (UInt)mIndices.size(), GL_UNSIGNED_INT, 0);
    }
    glBindVertexArray(0);
}