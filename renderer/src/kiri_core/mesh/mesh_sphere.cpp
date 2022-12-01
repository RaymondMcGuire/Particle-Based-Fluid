/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:49:16
 * @FilePath: \core\src\kiri_core\mesh\mesh_sphere.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_core/mesh/mesh_sphere.h>

void KiriMeshSphere::Construct()
{
    const float PI = kiri_math::pi<float>();
    mDrawElem = true;
    mVertDataType = DataType::Standard;
    for (Int y = 0; y <= Y_SEGMENTS; ++y)
    {
        for (Int x = 0; x <= X_SEGMENTS; ++x)
        {

            float xSegment = (float)x / (float)X_SEGMENTS;
            float ySegment = (float)y / (float)Y_SEGMENTS;

            float xPos = std::cos(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
            float yPos = std::cos(ySegment * PI);
            float zPos = std::sin(xSegment * 2.0f * PI) * std::sin(ySegment * PI);

            AddVertStand(Vector3F(xPos * mRadius, yPos * mRadius, zPos * mRadius), Vector3F(xPos, yPos, zPos), Vector2F(xSegment, ySegment));
        }
    }

    bool oddRow = false;
    for (Int y = 0; y < Y_SEGMENTS; ++y)
    {
        if (!oddRow) // even rows: y == 0, y == 2; and so on
        {
            for (Int x = 0; x <= X_SEGMENTS; ++x)
            {
                mIndices.append(y * (X_SEGMENTS + 1) + x);
                mIndices.append((y + 1) * (X_SEGMENTS + 1) + x);
            }
        }
        else
        {
            for (Int x = X_SEGMENTS; x >= 0; --x)
            {
                mIndices.append((y + 1) * (X_SEGMENTS + 1) + x);
                mIndices.append(y * (X_SEGMENTS + 1) + x);
            }
        }
        oddRow = !oddRow;
    }

    mVerticesNum = mVertStand.size();

    SetupVertex();
}

KiriMeshSphere::KiriMeshSphere()
{
    mInstance = false;

    Construct();
}

KiriMeshSphere::KiriMeshSphere(float _radius)
{
    mInstance = false;
    mRadius = _radius;
    Construct();
}

KiriMeshSphere::KiriMeshSphere(Array1<Matrix4x4F> _instMat4, bool _static_mesh)
{
    mInstance = true;
    mStaticMesh = _static_mesh;

    mInstMat4 = _instMat4;
    mInstanceType = 4;

    Construct();
}

KiriMeshSphere::KiriMeshSphere(float _radius, Int _xSeg, Int _ySeg, Array1<Matrix4x4F> _instMat4, bool _static_mesh)
{
    mInstance = true;
    mStaticMesh = _static_mesh;

    mRadius = _radius;
    X_SEGMENTS = _xSeg;
    Y_SEGMENTS = _ySeg;

    mInstMat4 = _instMat4;
    mInstanceType = 4;

    Construct();
}

void KiriMeshSphere::Draw()
{
    // KIRI_INFO << "Draw:" << mInstance;
    glBindVertexArray(mVAO);
    if (!mInstance)
    {
        glDrawElements(GL_TRIANGLE_STRIP, (UInt)mIndices.size(), GL_UNSIGNED_INT, 0);
    }
    else
    {
        glDrawElementsInstanced(GL_TRIANGLE_STRIP, (UInt)mIndices.size(), GL_UNSIGNED_INT, 0, (UInt)mInstMat4.size());
    }
    glBindVertexArray(0);
}