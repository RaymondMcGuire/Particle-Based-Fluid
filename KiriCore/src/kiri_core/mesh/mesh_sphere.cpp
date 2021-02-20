/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:33:39 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-18 22:00:05
 */
#include <kiri_core/mesh/mesh_sphere.h>

void KiriMeshSphere::Construct()
{
    const float PI = kiri_math::pi<float>();
    drawElem = true;
    vertDataType = DataType::Standard;
    for (Int y = 0; y <= Y_SEGMENTS; ++y)
    {
        for (Int x = 0; x <= X_SEGMENTS; ++x)
        {

            float xSegment = (float)x / (float)X_SEGMENTS;
            float ySegment = (float)y / (float)Y_SEGMENTS;

            float xPos = std::cos(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
            float yPos = std::cos(ySegment * PI);
            float zPos = std::sin(xSegment * 2.0f * PI) * std::sin(ySegment * PI);

            addVertStand(Vector3F(xPos * radius, yPos * radius, zPos * radius), Vector3F(xPos, yPos, zPos), Vector2F(xSegment, ySegment));
        }
    }

    bool oddRow = false;
    for (Int y = 0; y < Y_SEGMENTS; ++y)
    {
        if (!oddRow) // even rows: y == 0, y == 2; and so on
        {
            for (Int x = 0; x <= X_SEGMENTS; ++x)
            {
                indices.append(y * (X_SEGMENTS + 1) + x);
                indices.append((y + 1) * (X_SEGMENTS + 1) + x);
            }
        }
        else
        {
            for (Int x = X_SEGMENTS; x >= 0; --x)
            {
                indices.append((y + 1) * (X_SEGMENTS + 1) + x);
                indices.append(y * (X_SEGMENTS + 1) + x);
            }
        }
        oddRow = !oddRow;
    }

    verticesNum = vertStand.size();

    SetupVertex();
}

KiriMeshSphere::KiriMeshSphere()
{
    instancing = false;

    Construct();
}

KiriMeshSphere::KiriMeshSphere(float _radius)
{
    instancing = false;
    radius = _radius;
    Construct();
}

KiriMeshSphere::KiriMeshSphere(Array1<Matrix4x4F> _instMat4, bool _static_mesh)
{
    instancing = true;
    static_mesh = _static_mesh;

    instMat4 = _instMat4;
    instanceType = 4;

    Construct();
}

KiriMeshSphere::KiriMeshSphere(float _radius, Int _xSeg, Int _ySeg, Array1<Matrix4x4F> _instMat4, bool _static_mesh)
{
    instancing = true;
    static_mesh = _static_mesh;

    radius = _radius;
    X_SEGMENTS = _xSeg;
    Y_SEGMENTS = _ySeg;

    instMat4 = _instMat4;
    instanceType = 4;

    Construct();
}

void KiriMeshSphere::Draw()
{
    //KIRI_INFO << "Draw:" << instancing;
    glBindVertexArray(mVAO);
    if (!instancing)
    {
        glDrawElements(GL_TRIANGLE_STRIP, (UInt)indices.size(), GL_UNSIGNED_INT, 0);
    }
    else
    {
        glDrawElementsInstanced(GL_TRIANGLE_STRIP, (UInt)indices.size(), GL_UNSIGNED_INT, 0, (UInt)instMat4.size());
    }
    glBindVertexArray(0);
}