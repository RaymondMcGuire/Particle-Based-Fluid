/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:33:08 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-25 13:09:48
 */
#include <kiri_core/mesh/mesh_quad.h>
#include <glad/glad.h>
void KiriMeshQuad::Construct()
{
    drawElem = false;
    verticesNum = 4;

    if (imgMode)
    {
        vertDataType = DataType::Quad2;

        for (size_t i = 0; i < verticesNum; i++)
        {
            VertexQuad2 vs;
            vs.Position[0] = quadPos[i * 3 + 0];
            vs.Position[1] = quadPos[i * 3 + 1];
            vs.Position[2] = quadPos[i * 3 + 2];

            vs.TexCoords[0] = quadTexCoord[i * 2 + 0];
            vs.TexCoords[1] = quadTexCoord[i * 2 + 1];

            vertQuad2.append(vs);
        }
    }
    else
    {
        vertDataType = DataType::Quad3;
        for (size_t i = 0; i < verticesNum; i++)
        {
            VertexQuad3 vs;

            vs.Position[0] = quadPos[i * 3 + 0];
            vs.Position[1] = quadPos[i * 3 + 1];
            vs.Position[2] = quadPos[i * 3 + 2];

            vs.Color[0] = quadColor[i * 3 + 0];
            vs.Color[1] = quadColor[i * 3 + 1];
            vs.Color[2] = quadColor[i * 3 + 2];

            vs.TexCoords[0] = quadTexCoord[i * 2 + 0];
            vs.TexCoords[1] = quadTexCoord[i * 2 + 1];

            vertQuad3.append(vs);
        }
    }

    SetupVertex();
}

KiriMeshQuad::KiriMeshQuad()
{
    imgMode = true;
    instancing = false;

    Construct();
}

KiriMeshQuad::KiriMeshQuad(float _side)
{
    side = _side;
    imgMode = true;
    instancing = false;

    Array1<float> _quadPos = {
        -side, side, 0.0f,
        -side, -side, 0.0f,
        side, side, 0.0f,
        side, -side, 0.0f};

    quadPos = _quadPos;

    Construct();
}

KiriMeshQuad::KiriMeshQuad(float _side, Array1<Vector2F> _instVec2)
{
    side = _side;
    imgMode = false;
    instancing = true;

    Array1<float> _quadPos = {
        -side, side, 0.0f,
        -side, -side, 0.0f,
        side, side, 0.0f,
        side, -side, 0.0f};

    quadPos = _quadPos;

    instVec2 = _instVec2;
    instanceType = 2;

    Construct();
}

KiriMeshQuad::KiriMeshQuad(Array1<Vector2F> _instVec2)
{
    imgMode = false;
    instancing = true;

    instVec2 = _instVec2;
    instanceType = 2;

    Construct();
}

void KiriMeshQuad::Draw()
{
    glBindVertexArray(mVAO);
    if (!instancing)
    {
        glDrawArrays(GL_TRIANGLE_STRIP, 0, (UInt)verticesNum);
    }
    else
    {
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, (UInt)verticesNum, (UInt)instVec2.size());
    }

    glBindVertexArray(0);
}