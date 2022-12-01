/*
 * @Author: Xu.Wang
 * @Date: 2020-03-17 17:11:54
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-18 19:52:38
 */
#include <kiri_core/mesh/mesh_internal.h>
#include <glad/glad.h>
Array1<InstanceMat4x4> KiriMeshInternal::ConvertMat4ToFloatArray(Array1<Matrix4x4F> mat4)
{
    Array1<InstanceMat4x4> floatArray;

    for (size_t i = 0; i < mat4.size(); i++)
    {
        InstanceMat4x4 mat4x4;
        for (size_t j = 0; j < 16; j++)
        {
            mat4x4.value[j] = mat4.data()[i].data()[j];
        }

        floatArray.append(mat4x4);
    }

    return floatArray;
}

void KiriMeshInternal::SetupInstanceMat4(Int idx)
{
    glBindBuffer(GL_ARRAY_BUFFER, mInstanceVBO);
    glEnableVertexAttribArray(idx);
    glVertexAttribPointer(idx, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceMat4x4), (void *)0);
    glEnableVertexAttribArray(idx + 1);
    glVertexAttribPointer(idx + 1, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceMat4x4), (void *)(4 * sizeof(float)));
    glEnableVertexAttribArray(idx + 2);
    glVertexAttribPointer(idx + 2, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceMat4x4), (void *)(2 * 4 * sizeof(float)));
    glEnableVertexAttribArray(idx + 3);
    glVertexAttribPointer(idx + 3, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceMat4x4), (void *)(3 * 4 * sizeof(float)));

    glVertexAttribDivisor(idx, 1);
    glVertexAttribDivisor(idx + 1, 1);
    glVertexAttribDivisor(idx + 2, 1);
    glVertexAttribDivisor(idx + 3, 1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void KiriMeshInternal::SetupInstanceVec2(Int idx)
{
    glBindBuffer(GL_ARRAY_BUFFER, mInstanceVBO);

    glEnableVertexAttribArray(idx);
    glVertexAttribPointer(idx, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);

    glVertexAttribDivisor(2, 1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void KiriMeshInternal::InitInstanceVBO(Int type)
{
    switch (type)
    {
    case 2:
        InitInstanceVec2(mInstVec2);
        SetupInstanceVec2(mInstanceVertNum);
        break;
    case 4:
        InitInstanceMat4(mInstMat4);
        SetupInstanceMat4(mInstanceVertNum);
        break;

    default:
        break;
    }
}

void KiriMeshInternal::UpdateInstance(Array1Mat4x4F instMat4)
{
    mInstMat4 = instMat4;

    if (!mStaticMesh)
    {
        Array1<InstanceMat4x4> fA = ConvertMat4ToFloatArray(mInstMat4);
        glBindBuffer(GL_ARRAY_BUFFER, mInstanceVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(InstanceMat4x4) * mInstMat4.size(), NULL, GL_STREAM_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(InstanceMat4x4) * mInstMat4.size(), fA.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void KiriMeshInternal::InitInstanceVec2(Array1<Vector2F> instVec2)
{
    glGenBuffers(1, &mInstanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, mInstanceVBO);
    if (mStaticMesh)
        glBufferData(GL_ARRAY_BUFFER, 2 * sizeof(float) * MAX_INSTANCE_NUM, instVec2.data(), GL_STATIC_DRAW);
    else
        glBufferData(GL_ARRAY_BUFFER, 2 * sizeof(float) * MAX_INSTANCE_NUM, NULL, GL_STREAM_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void KiriMeshInternal::InitInstanceMat4(Array1<Matrix4x4F> instMat4)
{
    Array1<InstanceMat4x4> fA = ConvertMat4ToFloatArray(instMat4);
    glGenBuffers(1, &mInstanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, mInstanceVBO);
    if (mStaticMesh)
        glBufferData(GL_ARRAY_BUFFER, sizeof(InstanceMat4x4) * instMat4.size(), fA.data(), GL_STATIC_DRAW);
    else
        glBufferData(GL_ARRAY_BUFFER, sizeof(InstanceMat4x4) * instMat4.size(), NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void KiriMeshInternal::SetupVertex()
{
    // mVAO
    glGenVertexArrays(1, &mVAO);
    glBindVertexArray(mVAO);

    // mVBO
    glGenBuffers(1, &mVBO);
    glBindBuffer(GL_ARRAY_BUFFER, mVBO);

    switch (mVertDataType)
    {
    case Simple:
        glBufferData(GL_ARRAY_BUFFER, mVertSimple.size() * sizeof(VertexSimple), mVertSimple.data(), GL_STATIC_DRAW);
        break;
    case Standard:
        glBufferData(GL_ARRAY_BUFFER, mVertStand.size() * sizeof(VertexStandard), mVertStand.data(), GL_STATIC_DRAW);
        break;
    case Full:
        glBufferData(GL_ARRAY_BUFFER, mVertFull.size() * sizeof(VertexFull), mVertFull.data(), GL_STATIC_DRAW);
        break;
    case Quad2:
        glBufferData(GL_ARRAY_BUFFER, mVertQuad2.size() * sizeof(VertexQuad2), mVertQuad2.data(), GL_STATIC_DRAW);
        break;
    case Quad3:
        glBufferData(GL_ARRAY_BUFFER, mVertQuad3.size() * sizeof(VertexQuad3), mVertQuad3.data(), GL_STATIC_DRAW);
        break;
    default:
        break;
    }

    // mEBO
    if (mDrawElem)
    {
        glGenBuffers(1, &mEBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mIndices.size() * sizeof(UInt), mIndices.data(), GL_STATIC_DRAW);
    }

    switch (mVertDataType)
    {
    case Simple:
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, POSITION_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexSimple), (void *)0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, COLOR_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexSimple), (void *)offsetof(VertexSimple, Color));

        break;
    case Standard:
        mInstanceVertNum = 3;
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, POSITION_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexStandard), (void *)0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, NORMAL_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexStandard), (void *)offsetof(VertexStandard, Normal));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, TEXCOORD_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexStandard), (void *)offsetof(VertexStandard, TexCoords));
        break;
    case Full:
        mInstanceVertNum = 5;
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, POSITION_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexFull), (void *)0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, NORMAL_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexFull), (void *)offsetof(VertexFull, Normal));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, TEXCOORD_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexFull), (void *)offsetof(VertexFull, TexCoords));

        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, TANGENT_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexFull), (void *)offsetof(VertexFull, Tangent));

        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, BITANGENT_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexFull), (void *)offsetof(VertexFull, Bitangent));
        break;
    case Quad2:
        mInstanceVertNum = 2;
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, POSITION_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexQuad2), (void *)0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, TEXCOORD_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexQuad2), (void *)offsetof(VertexQuad2, TexCoords));

        break;
    case Quad3:
        mInstanceVertNum = 3;
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, POSITION_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexQuad3), (void *)0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, COLOR_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexQuad3), (void *)offsetof(VertexQuad3, Color));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, TEXCOORD_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexQuad3), (void *)offsetof(VertexQuad3, TexCoords));
        break;
    default:
        break;
    }

    // mInstanceVBO
    if (mInstance)
    {
        InitInstanceVBO(mInstanceType);
    }

    glBindVertexArray(0);
}

void KiriMeshInternal::AddVertSimple(Vector3F pos, Vector3F col)
{
    VertexSimple vs;
    vs.Position[0] = pos.x;
    vs.Position[1] = pos.y;
    vs.Position[2] = pos.z;

    vs.Color[0] = col.x;
    vs.Color[1] = col.y;
    vs.Color[2] = col.z;
    mVertSimple.append(vs);

    mVerticesNum++;
}

void KiriMeshInternal::AddVertStand(Vector3F pos, Vector3F norm, Vector2F tex)
{
    VertexStandard vs;
    vs.Position[0] = pos.x;
    vs.Position[1] = pos.y;
    vs.Position[2] = pos.z;

    vs.Normal[0] = norm.x;
    vs.Normal[1] = norm.y;
    vs.Normal[2] = norm.z;

    vs.TexCoords[0] = tex.x;
    vs.TexCoords[1] = tex.y;

    mVertStand.append(vs);
}