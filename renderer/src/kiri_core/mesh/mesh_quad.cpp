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
    mDrawElem = false;
    mVerticesNum = 4;

    if (mImgMode)
    {
        mVertDataType = DataType::Quad2;

        for (size_t i = 0; i < mVerticesNum; i++)
        {
            VertexQuad2 vs;
            vs.Position[0] = mQuadPos[i * 3 + 0];
            vs.Position[1] = mQuadPos[i * 3 + 1];
            vs.Position[2] = mQuadPos[i * 3 + 2];

            vs.TexCoords[0] = mQuadTexCoord[i * 2 + 0];
            vs.TexCoords[1] = mQuadTexCoord[i * 2 + 1];

            mVertQuad2.append(vs);
        }
    }
    else
    {
        mVertDataType = DataType::Quad3;
        for (size_t i = 0; i < mVerticesNum; i++)
        {
            VertexQuad3 vs;

            vs.Position[0] = mQuadPos[i * 3 + 0];
            vs.Position[1] = mQuadPos[i * 3 + 1];
            vs.Position[2] = mQuadPos[i * 3 + 2];

            vs.Color[0] = mQuadColor[i * 3 + 0];
            vs.Color[1] = mQuadColor[i * 3 + 1];
            vs.Color[2] = mQuadColor[i * 3 + 2];

            vs.TexCoords[0] = mQuadTexCoord[i * 2 + 0];
            vs.TexCoords[1] = mQuadTexCoord[i * 2 + 1];

            mVertQuad3.append(vs);
        }
    }

    SetupVertex();
}

KiriMeshQuad::KiriMeshQuad()
{
    mImgMode = true;
    mInstance = false;

    Construct();
}

KiriMeshQuad::KiriMeshQuad(float _side)
{
    mSide = _side;
    mImgMode = true;
    mInstance = false;

    Array1<float> _quadPos = {
        -mSide, mSide, 0.0f,
        -mSide, -mSide, 0.0f,
        mSide, mSide, 0.0f,
        mSide, -mSide, 0.0f};

    mQuadPos = _quadPos;

    Construct();
}

KiriMeshQuad::KiriMeshQuad(float _side, Array1<Vector2F> _instVec2)
{
    mSide = _side;
    mImgMode = false;
    mInstance = true;

    Array1<float> _quadPos = {
        -mSide, mSide, 0.0f,
        -mSide, -mSide, 0.0f,
        mSide, mSide, 0.0f,
        mSide, -mSide, 0.0f};

    mQuadPos = _quadPos;

    mInstVec2 = _instVec2;
    mInstanceType = 2;

    Construct();
}

KiriMeshQuad::KiriMeshQuad(Array1<Vector2F> _instVec2)
{
    mImgMode = false;
    mInstance = true;

    mInstVec2 = _instVec2;
    mInstanceType = 2;

    Construct();
}

void KiriMeshQuad::Draw()
{
    glBindVertexArray(mVAO);
    if (!mInstance)
    {
        glDrawArrays(GL_TRIANGLE_STRIP, 0, (UInt)mVerticesNum);
    }
    else
    {
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, (UInt)mVerticesNum, (UInt)mInstVec2.size());
    }

    glBindVertexArray(0);
}