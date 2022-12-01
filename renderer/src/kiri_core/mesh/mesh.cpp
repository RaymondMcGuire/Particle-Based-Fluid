/*
 * @Author: Xu.Wang
 * @Date: 2020-03-17 17:34:11
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-25 18:00:07
 */
#include <kiri_core/mesh/mesh.h>
#include <glad/glad.h>
KiriMesh::KiriMesh()
{
    mInstance = false;
    mVAO = mVBO = mEBO = 0;
    type = DataType::Standard;
}

KiriMesh::KiriMesh(Array1<VertexSimple> vs)
{
    mInstance = false;
    mVAO = mVBO = mEBO = 0;
    type = DataType::Standard;
    this->mVertSimple = vs;
    SetupSimple();
}

KiriMesh::KiriMesh(Array1<VertexStandard> vs, Array1<UInt> _indices, bool _instance, Array1<Matrix4x4F> _trans4)
{
    mInstance = _instance;
    trans4 = _trans4;
    mInstMat4(trans4);

    mVAO = mVBO = mEBO = 0;
    type = DataType::Standard;
    this->mVertStand = vs;
    this->mIndices = _indices;
    SetupStand();
}

KiriMesh::KiriMesh(Array1<VertexFull> vf, Array1<UInt> _indices, bool _instance, Array1<Matrix4x4F> _trans4)
{
    mInstance = _instance;
    trans4 = _trans4;
    mInstMat4(trans4);

    mVAO = mVBO = mEBO = 0;
    type = DataType::Full;
    this->mVertFull = vf;
    this->mIndices = _indices;
    SetupFull();
}

KiriMesh::KiriMesh(Array1<VertexStandard> vs, Array1<UInt> _indices, Array1<Texture> _textures, bool _instance, Array1<Matrix4x4F> _trans4)
{
    mInstance = _instance;
    trans4 = _trans4;
    mInstMat4(trans4);

    mVAO = mVBO = mEBO = 0;
    type = DataType::Standard;
    this->mVertStand = vs;
    this->mIndices = _indices;
    this->mTextures = _textures;
    SetupStand();
}

KiriMesh::KiriMesh(Array1<VertexFull> vf, Array1<UInt> _indices, Array1<Texture> _textures, bool _instance, Array1<Matrix4x4F> _trans4)
{
    mInstance = _instance;
    trans4 = _trans4;
    mInstMat4(trans4);

    mVAO = mVBO = mEBO = 0;
    type = DataType::Full;
    this->mVertFull = vf;
    this->mIndices = _indices;
    this->mTextures = _textures;
    SetupFull();
}

void KiriMesh::SetupInstance(Int idx)
{
    if (mInstance)
    {
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);

        glEnableVertexAttribArray(idx);
        glVertexAttribPointer(idx, 4, GL_FLOAT, GL_FALSE, sizeof(Matrix4x4F), (void *)0);
        glEnableVertexAttribArray(idx + 1);
        glVertexAttribPointer(idx + 1, 4, GL_FLOAT, GL_FALSE, sizeof(Matrix4x4F), (void *)(sizeof(Vector4F)));
        glEnableVertexAttribArray(idx + 2);
        glVertexAttribPointer(idx + 2, 4, GL_FLOAT, GL_FALSE, sizeof(Matrix4x4F), (void *)(2 * sizeof(Vector4F)));
        glEnableVertexAttribArray(idx + 3);
        glVertexAttribPointer(idx + 3, 4, GL_FLOAT, GL_FALSE, sizeof(Matrix4x4F), (void *)(3 * sizeof(Vector4F)));

        glVertexAttribDivisor(idx, 1);
        glVertexAttribDivisor(idx + 1, 1);
        glVertexAttribDivisor(idx + 2, 1);
        glVertexAttribDivisor(idx + 3, 1);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void KiriMesh::SetupSimple()
{
    // create
    glGenVertexArrays(1, &mVAO);
    glGenBuffers(1, &mVBO);

    // Bind
    glBindVertexArray(mVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mVBO);

    // vertices
    glBufferData(GL_ARRAY_BUFFER, (GLsizei)mVertSimple.size() * sizeof(VertexSimple), mVertSimple.data(), GL_STATIC_DRAW);

    // vertex position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexSimple), (void *)0);
    glBindVertexArray(0);
}

void KiriMesh::SetupStand()
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

    SetupInstance(3);

    glBindVertexArray(0);
}

void KiriMesh::SetupFull()
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

    SetupInstance(5);

    glBindVertexArray(0);
}

void KiriMesh::Draw(KiriShader *shader)
{
    // Bind appropriate mTextures
    UInt diffuseNr = 1;
    UInt specularNr = 1;
    UInt normalNr = 1;
    UInt heightNr = 1;

    for (UInt i = 0; i < mTextures.size(); i++)
    {
        glActiveTexture(GL_TEXTURE0 + i); // active proper texture unit before binding
        // retrieve texture number (the N in diffuse_textureN)
        String number;
        String name = mTextures[i].type;
        if (name == "texture_diffuse")
            number = std::to_string(diffuseNr++);
        else if (name == "texture_specular")
            number = std::to_string(specularNr++); // transfer UInt to stream
        else if (name == "texture_normal")
            number = std::to_string(normalNr++); // transfer UInt to stream
        else if (name == "texture_height")
            number = std::to_string(heightNr++); // transfer UInt to stream

        // now set the sampler to the correct texture unit
        glUniform1i(glGetUniformLocation(shader->ID, (name + number).c_str()), i);
        // and finally Bind the texture
        glBindTexture(GL_TEXTURE_2D, mTextures[i].id);
    }

    // Draw mesh
    glBindVertexArray(mVAO);
    if (!mInstance)
    {
        glDrawElements(GL_TRIANGLES, (GLsizei)mIndices.size(), GL_UNSIGNED_INT, 0);
    }
    else
    {
        glDrawElementsInstanced(GL_TRIANGLES, (GLsizei)mIndices.size(), GL_UNSIGNED_INT, 0, (GLsizei)trans4.size());
    }

    glBindVertexArray(0);
    glActiveTexture(GL_TEXTURE0);
}

void KiriMesh::mInstMat4(Array1<Matrix4x4F> _trans4)
{
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, 16 * sizeof(float) * (GLsizei)_trans4.size(), _trans4.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

KiriMesh::~KiriMesh()
{
    glDeleteVertexArrays(1, &mVAO);
    glDeleteBuffers(1, &mVBO);
    glDeleteBuffers(1, &mEBO);

    if (mInstance)
        glDeleteBuffers(1, &instanceVBO);
}