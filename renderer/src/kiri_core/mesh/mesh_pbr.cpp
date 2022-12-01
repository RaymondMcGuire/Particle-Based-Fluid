/*
 * @Author: Xu.Wang
 * @Date: 2020-03-17 17:34:06
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-25 17:59:51
 */
#include <kiri_core/mesh/mesh_pbr.h>
#include <glad/glad.h>
KiriMeshPBR::KiriMeshPBR()
{
    mVAO = mVBO = mEBO = 0;
    mDataType = DataType::Standard;
}

KiriMeshPBR::KiriMeshPBR(Array1<VertexStandard> vs, Array1<UInt> _indices)
{
    mVAO = mVBO = mEBO = 0;
    mDataType = DataType::Standard;
    this->mVertStand = vs;
    this->mIndices = _indices;
    SetupStand();
}

KiriMeshPBR::KiriMeshPBR(Array1<VertexFull> vf, Array1<UInt> _indices)
{
    mVAO = mVBO = mEBO = 0;
    mDataType = DataType::Full;
    this->mVertFull = vf;
    this->mIndices = _indices;
    SetupFull();
}

KiriMeshPBR::KiriMeshPBR(Array1<VertexStandard> vs, Array1<UInt> _indices, Array1<Texture> _textures)
{
    mVAO = mVBO = mEBO = 0;
    mDataType = DataType::Standard;
    this->mVertStand = vs;
    this->mIndices = _indices;
    this->mTextures = _textures;
    SetupStand();
}

KiriMeshPBR::KiriMeshPBR(Array1<VertexFull> vf, Array1<UInt> _indices, Array1<Texture> _textures)
{
    mVAO = mVBO = mEBO = 0;
    mDataType = DataType::Full;
    this->mVertFull = vf;
    this->mIndices = _indices;
    this->mTextures = _textures;
    SetupFull();
}

void KiriMeshPBR::SetupStand()
{
    // create
    glGenVertexArrays(1, &mVAO);
    glGenBuffers(1, &mVBO);
    glGenBuffers(1, &mEBO);

    // Bind
    glBindVertexArray(mVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mVBO);

    // vertices
    glBufferData(GL_ARRAY_BUFFER, (GLsizei)mVertStand.size() * sizeof(VertexStandard), mVertStand.data(), GL_STATIC_DRAW);

    // element Draw
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizei)mIndices.size() * sizeof(UInt), mIndices.data(), GL_STATIC_DRAW);

    // vertex position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexStandard), (void *)0);
    // vertex normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexStandard), (void *)offsetof(VertexStandard, Normal));
    // vertex texture coords
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexStandard), (void *)offsetof(VertexStandard, TexCoords));

    glBindVertexArray(0);
}

void KiriMeshPBR::SetupFull()
{
    // create
    glGenVertexArrays(1, &mVAO);
    glGenBuffers(1, &mVBO);
    glGenBuffers(1, &mEBO);

    // Bind
    glBindVertexArray(mVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mVBO);

    // vertices
    glBufferData(GL_ARRAY_BUFFER, (GLsizei)mVertFull.size() * sizeof(VertexFull), mVertFull.data(), GL_STATIC_DRAW);

    // element Draw
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizei)mIndices.size() * sizeof(UInt), mIndices.data(), GL_STATIC_DRAW);

    // vertex position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexFull), (void *)0);
    // vertex normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexFull), (void *)offsetof(VertexFull, Normal));
    // vertex texture coords
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexFull), (void *)offsetof(VertexFull, TexCoords));
    // vertex tangent
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(VertexFull), (void *)offsetof(VertexFull, Tangent));
    // vertex bitangent
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(VertexFull), (void *)offsetof(VertexFull, Bitangent));

    glBindVertexArray(0);
}

void KiriMeshPBR::Draw()
{
    glBindVertexArray(mVAO);
    glDrawElements(GL_TRIANGLES, (GLsizei)mIndices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    glActiveTexture(GL_TEXTURE0);
}

KiriMeshPBR::~KiriMeshPBR()
{
    glDeleteVertexArrays(1, &mVAO);
    glDeleteBuffers(1, &mVBO);
    glDeleteBuffers(1, &mEBO);
}