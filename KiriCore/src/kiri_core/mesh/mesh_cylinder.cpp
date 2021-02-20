/*
 * @Author: Song Ho Ahn 
 * @Url: http://www.songho.ca/opengl/gl_cylinder.html
 * @Date: 2020-03-25 11:14:08 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-25 17:58:14
 */

#include <kiri_core/mesh/mesh_cylinder.h>

///////////////////////////////////////////////////////////////////////////////
// generate 3D vertices of a unit circle on XY plance
///////////////////////////////////////////////////////////////////////////////
void KiriMeshCylinder::buildUnitCircleVertices()
{
    float sectorStep = 2 * kiri_math::pi<float>() / sectorCount;
    float sectorAngle; // radian

    unitCircleVertices.clear();
    for (Int i = 0; i <= sectorCount; ++i)
    {
        sectorAngle = i * sectorStep;
        unitCircleVertices.append(cos(sectorAngle)); // x
        unitCircleVertices.append(sin(sectorAngle)); // y
        unitCircleVertices.append(0);                // z
    }
}

///////////////////////////////////////////////////////////////////////////////
// generate shared normal vectors of the side of cylinder
///////////////////////////////////////////////////////////////////////////////
Array1<float> KiriMeshCylinder::getSideNormals()
{
    float sectorStep = 2 * kiri_math::pi<float>() / sectorCount;
    float sectorAngle; // radian

    // compute the normal vector at 0 degree first
    // tanA = (baseRadius-topRadius) / height
    float zAngle = atan2(baseRadius - topRadius, height);
    float x0 = cos(zAngle); // nx
    float y0 = 0;           // ny
    float z0 = sin(zAngle); // nz

    // Rotate (x0,y0,z0) per sector angle
    Array1<float> normals;
    for (Int i = 0; i <= sectorCount; ++i)
    {
        sectorAngle = i * sectorStep;
        normals.append(cos(sectorAngle) * x0 - sin(sectorAngle) * y0); // nx
        normals.append(sin(sectorAngle) * x0 + cos(sectorAngle) * y0); // ny
        normals.append(z0);                                            // nz
    }

    return normals;
}

void KiriMeshCylinder::buildVerticesSmooth()
{
    float x, y, z; // vertex position
    //float s, t;                                     // texCoord
    float radius; // radius for each stack

    // get normals for cylinder sides
    Array1<float> sideNormals = getSideNormals();

    // put vertices of side cylinder to array by scaling unit circle
    for (Int i = 0; i <= stackCount; ++i)
    {
        z = -(height * 0.5f) + (float)i / stackCount * height;                  // vertex position z
        radius = baseRadius + (float)i / stackCount * (topRadius - baseRadius); // lerp
        float t = 1.0f - (float)i / stackCount;                                 // top-to-bottom

        for (Int j = 0, k = 0; j <= sectorCount; ++j, k += 3)
        {
            x = unitCircleVertices[k];
            y = unitCircleVertices[k + 1];

            addVertStand(Vector3F(x * radius, y * radius, z), Vector3F(sideNormals[k], sideNormals[k + 1], sideNormals[k + 2]), Vector2F((float)j / sectorCount, t));
        }
    }

    // remember where the base.top vertices start
    UInt baseVertexIndex = (UInt)vertStand.size();

    // put vertices of base of cylinder
    z = -height * 0.5f;
    addVertStand(Vector3F(0, 0, z), Vector3F(0, 0, -1), Vector2F(0.5f, 0.5f));
    for (Int i = 0, j = 0; i < sectorCount; ++i, j += 3)
    {
        x = unitCircleVertices[j];
        y = unitCircleVertices[j + 1];
        // texcoord flip horizontal
        addVertStand(Vector3F(x * baseRadius, y * baseRadius, z), Vector3F(0, 0, -1), Vector2F(-x * 0.5f + 0.5f, -y * 0.5f + 0.5f));
    }

    // remember where the base vertices start
    UInt topVertexIndex = (UInt)vertStand.size();

    // put vertices of top of cylinder
    z = height * 0.5f;
    addVertStand(Vector3F(0, 0, z), Vector3F(0, 0, 1), Vector2F(0.5f, 0.5f));
    for (Int i = 0, j = 0; i < sectorCount; ++i, j += 3)
    {
        x = unitCircleVertices[j];
        y = unitCircleVertices[j + 1];
        addVertStand(Vector3F(x * topRadius, y * topRadius, z), Vector3F(0, 0, 1), Vector2F(x * 0.5f + 0.5f, y * 0.5f + 0.5f));
    }

    // put indices for sides
    UInt k1, k2;
    for (Int i = 0; i < stackCount; ++i)
    {
        k1 = i * (sectorCount + 1); // bebinning of current stack
        k2 = k1 + sectorCount + 1;  // beginning of next stack

        for (Int j = 0; j < sectorCount; ++j, ++k1, ++k2)
        {
            // 2 trianles per sector
            indices.append(k1);
            indices.append(k1 + 1);
            indices.append(k2);

            indices.append(k2);
            indices.append(k1 + 1);
            indices.append(k2 + 1);

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

    // remember where the base indices start
    baseIndex = (UInt)indices.size();

    // put indices for base
    for (Int i = 0, k = baseVertexIndex + 1; i < sectorCount; ++i, ++k)
    {
        if (i < (sectorCount - 1))
        {
            indices.append(baseVertexIndex);
            indices.append(k + 1);
            indices.append(k);
        }
        else
        {
            indices.append(baseVertexIndex);
            indices.append(baseVertexIndex + 1);
            indices.append(k);
        }
    }

    // remember where the base indices start
    topIndex = (UInt)indices.size();

    for (Int i = 0, k = topVertexIndex + 1; i < sectorCount; ++i, ++k)
    {
        if (i < (sectorCount - 1))
        {
            indices.append(topVertexIndex);
            indices.append(k);
            indices.append(k + 1);
        }
        else
        {
            indices.append(topVertexIndex);
            indices.append(k);
            indices.append(topVertexIndex + 1);
        }
    }
}

void KiriMeshCylinder::Construct()
{
    drawElem = true;
    vertDataType = DataType::Standard;

    buildUnitCircleVertices();

    if (smooth)
    {
        buildVerticesSmooth();
    }

    verticesNum = vertStand.size();

    SetupVertex();
}

KiriMeshCylinder::KiriMeshCylinder(float baseRadius, float topRadius, float height, Int sectors,
                                   Int stacks, bool smooth)
{
    instancing = false;

    this->baseRadius = baseRadius;
    this->topRadius = topRadius;
    this->height = height;
    this->sectorCount = sectors;
    if (sectors < MIN_SECTOR_COUNT)
        this->sectorCount = MIN_SECTOR_COUNT;
    this->stackCount = stacks;
    if (stacks < MIN_STACK_COUNT)
        this->stackCount = MIN_STACK_COUNT;
    this->smooth = smooth;

    Construct();
}

void KiriMeshCylinder::Draw()
{

    glBindVertexArray(mVAO);
    if (!instancing)
    {
        glDrawElements(GL_TRIANGLES, (UInt)indices.size(), GL_UNSIGNED_INT, 0);
    }
    // else
    // {
    //     // not test
    //     glDrawElementsInstanced(GL_TRIANGLES, (UInt)indices.size(), GL_UNSIGNED_INT, indices.data(), (UInt)instMat4.size());
    // }
    glBindVertexArray(0);
}