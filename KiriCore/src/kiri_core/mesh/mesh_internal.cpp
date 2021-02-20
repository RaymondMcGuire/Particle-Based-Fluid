/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:11:54 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-18 19:52:38
 */
#include <kiri_core/mesh/mesh_internal.h>
#include <glad/glad.h>
Array1<InstanceMat4x4> KiriMeshInternal::cvtMat4ToFloatArray(Array1<Matrix4x4F> mat4)
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
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
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
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);

    glEnableVertexAttribArray(idx);
    glVertexAttribPointer(idx, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);

    glVertexAttribDivisor(2, 1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void KiriMeshInternal::initInstanceVBO(Int type)
{
    switch (type)
    {
    case 2:
        initInstanceVec2(instVec2);
        SetupInstanceVec2(instanceVertNum);
        break;
    case 4:
        initInstanceMat4(instMat4);
        SetupInstanceMat4(instanceVertNum);
        break;

    default:
        break;
    }
}

void KiriMeshInternal::UpdateInstance(Array1Mat4x4F _instMat4)
{
    instMat4 = _instMat4;

    if (!static_mesh)
    {
        Array1<InstanceMat4x4> fA = cvtMat4ToFloatArray(instMat4);
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(InstanceMat4x4) * instMat4.size(), NULL, GL_STREAM_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(InstanceMat4x4) * instMat4.size(), fA.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void KiriMeshInternal::initInstanceVec2(Array1<Vector2F> _instVec2)
{
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    if (static_mesh)
        glBufferData(GL_ARRAY_BUFFER, 2 * sizeof(float) * MAX_INSTANCE_NUM, _instVec2.data(), GL_STATIC_DRAW);
    else
        glBufferData(GL_ARRAY_BUFFER, 2 * sizeof(float) * MAX_INSTANCE_NUM, NULL, GL_STREAM_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void KiriMeshInternal::initInstanceMat4(Array1<Matrix4x4F> _instMat4)
{
    Array1<InstanceMat4x4> fA = cvtMat4ToFloatArray(_instMat4);
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    if (static_mesh)
        glBufferData(GL_ARRAY_BUFFER, sizeof(InstanceMat4x4) * _instMat4.size(), fA.data(), GL_STATIC_DRAW);
    else
        glBufferData(GL_ARRAY_BUFFER, sizeof(InstanceMat4x4) * _instMat4.size(), NULL, GL_STREAM_DRAW);
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

    switch (vertDataType)
    {
    case Standard:
        glBufferData(GL_ARRAY_BUFFER, vertStand.size() * sizeof(VertexStandard), vertStand.data(), GL_STATIC_DRAW);
        break;
    case Full:
        glBufferData(GL_ARRAY_BUFFER, vertFull.size() * sizeof(VertexFull), vertFull.data(), GL_STATIC_DRAW);
        break;
    case Quad2:
        glBufferData(GL_ARRAY_BUFFER, vertQuad2.size() * sizeof(VertexQuad2), vertQuad2.data(), GL_STATIC_DRAW);
        break;
    case Quad3:
        glBufferData(GL_ARRAY_BUFFER, vertQuad3.size() * sizeof(VertexQuad3), vertQuad3.data(), GL_STATIC_DRAW);
        break;
    default:
        break;
    }

    // mEBO
    if (drawElem)
    {
        glGenBuffers(1, &mEBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(UInt), indices.data(), GL_STATIC_DRAW);
    }

    switch (vertDataType)
    {
    case Standard:
        instanceVertNum = 3;
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, POSITION_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexStandard), (void *)0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, NORMAL_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexStandard), (void *)offsetof(VertexStandard, Normal));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, TEXCOORD_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexStandard), (void *)offsetof(VertexStandard, TexCoords));
        break;
    case Full:
        instanceVertNum = 5;
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
        instanceVertNum = 2;
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, POSITION_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexQuad2), (void *)0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, TEXCOORD_LENGTH, GL_FLOAT, GL_FALSE, sizeof(VertexQuad2), (void *)offsetof(VertexQuad2, TexCoords));

        break;
    case Quad3:
        instanceVertNum = 3;
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

    // instanceVBO
    if (instancing)
    {
        initInstanceVBO(instanceType);
    }

    glBindVertexArray(0);
}

void KiriMeshInternal::addVertStand(Vector3F _pos, Vector3F _norm, Vector2F _tex)
{
    VertexStandard vs;
    vs.Position[0] = _pos.x;
    vs.Position[1] = _pos.y;
    vs.Position[2] = _pos.z;

    vs.Normal[0] = _norm.x;
    vs.Normal[1] = _norm.y;
    vs.Normal[2] = _norm.z;

    vs.TexCoords[0] = _tex.x;
    vs.TexCoords[1] = _tex.y;

    vertStand.append(vs);
}