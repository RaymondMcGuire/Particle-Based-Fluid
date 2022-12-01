/*
 * @Author: Xu.Wang
 * @Date: 2020-03-17 17:35:24
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-20 22:49:16
 */
#include <kiri_core/model/model_load_obj.h>
#include <kiri_utils.h>
void KiriModelLoadObj::Draw()
{
    if (!mInstance)
    {
        KiriModel::Draw();
    }

    for (UInt i = 0; i < meshes.size(); i++)
        meshes[i]->Draw(mMat->GetShader());
}

void KiriModelLoadObj::Load(String const &path)
{
    // read file via ASSIMP
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
    // check for errors
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
    {
        std::cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
        return;
    }
    // retrieve the directory path of the filepath
    directory = path.substr(0, path.find_last_of('/'));

    // process ASSIMP's root node recursively
    ProcessNode(scene->mRootNode, scene);
}

void KiriModelLoadObj::ProcessNode(aiNode *node, const aiScene *scene)
{
    // process each mesh located at the current node
    for (UInt i = 0; i < node->mNumMeshes; i++)
    {
        // the node object only contains mIndices to index the actual objects in the scene.
        // the scene contains all the data, node is just to keep stuff organized (like relations between nodes).
        aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.append(ProcessMesh(mesh, scene));
    }
    // after we've processed all of the meshes (if any) we then recursively process each of the children nodes
    for (UInt i = 0; i < node->mNumChildren; i++)
    {
        ProcessNode(node->mChildren[i], scene);
    }
}

KiriMesh *KiriModelLoadObj::ProcessMesh(aiMesh *mesh, const aiScene *scene)
{
    Array1<VertexFull> vertices;
    Array1<UInt> mIndices;
    Array1<Texture> mTextures;

    for (UInt i = 0; i < mesh->mNumVertices; i++)
    {
        VertexFull vertex;

        // positions
        vertex.Position[0] = mesh->mVertices[i].x;
        vertex.Position[1] = mesh->mVertices[i].y;
        vertex.Position[2] = mesh->mVertices[i].z;

        // normals
        vertex.Normal[0] = mesh->mNormals[i].x;
        vertex.Normal[1] = mesh->mNormals[i].y;
        vertex.Normal[2] = mesh->mNormals[i].z;

        // texture coordinates
        if (mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
        {
            // a vertex can contain up to 8 different texture coordinates. We thus make the assumption that we won't
            // use models where a vertex can have multiple texture coordinates so we always take the first set (0).
            vertex.TexCoords[0] = mesh->mTextureCoords[0][i].x;
            vertex.TexCoords[1] = mesh->mTextureCoords[0][i].y;
        }
        else
        {
            vertex.TexCoords[0] = 0.0f;
            vertex.TexCoords[1] = 0.0f;
        }
        // tangent
        vertex.Tangent[0] = mesh->mTangents[i].x;
        vertex.Tangent[1] = mesh->mTangents[i].y;
        vertex.Tangent[2] = mesh->mTangents[i].z;

        // bitangent
        vertex.Bitangent[0] = mesh->mBitangents[i].x;
        vertex.Bitangent[1] = mesh->mBitangents[i].y;
        vertex.Bitangent[2] = mesh->mBitangents[i].z;

        vertices.append(vertex);
    }
    // now wak through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex mIndices.
    for (UInt i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];
        // retrieve all mIndices of the face and store them in the mIndices vector
        for (UInt j = 0; j < face.mNumIndices; j++)
            mIndices.append(face.mIndices[j]);
    }
    // process materials
    aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];
    // we assume a convention for sampler names in the shaders. Each mDiffuse texture should be named
    // as 'texture_diffuseN' where N is a sequential number ranging from 1 to MAX_SAMPLER_NUMBER.
    // Same applies to other texture as the following list summarizes:
    // mDiffuse: texture_diffuseN
    // specular: texture_specularN
    // mNormal: texture_normalN

    // 1. mDiffuse maps
    Array1<Texture> diffuseMaps = LoadMaterialTextures(material, aiTextureType_DIFFUSE, "texture_diffuse");
    mTextures.append(diffuseMaps);
    // 2. specular maps
    Array1<Texture> specularMaps = LoadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular");
    mTextures.append(specularMaps);
    // 3. mNormal maps
    Array1<Texture> normalMaps = LoadMaterialTextures(material, aiTextureType_HEIGHT, "texture_normal");
    mTextures.append(normalMaps);
    // 4. height maps
    Array1<Texture> heightMaps = LoadMaterialTextures(material, aiTextureType_AMBIENT, "texture_height");
    mTextures.append(heightMaps);

    // return a mesh object created from the extracted mesh data
    return new KiriMesh(vertices, mIndices, mTextures, mInstance, trans4);
}

Array1<Texture> KiriModelLoadObj::LoadMaterialTextures(aiMaterial *mat, aiTextureType type, String typeName)
{
    Array1<Texture> mTextures;
    for (UInt i = 0; i < mat->GetTextureCount(type); i++)
    {
        aiString str;
        mat->GetTexture(type, i, &str);
        // check if texture was loaded before and if so, continue to next iteration: skip loading a new texture
        bool skip = false;
        for (UInt j = 0; j < textures_loaded.size(); j++)
        {
            if (std::strcmp(textures_loaded[j].path.data(), str.C_Str()) == 0)
            {
                mTextures.append(textures_loaded[j]);
                skip = true; // a texture with the same filepath has already been loaded, continue to next one. (optimization)
                break;
            }
        }
        if (!skip)
        { // if texture hasn't been loaded already, load it
            Texture texture;
            texture.id = KiriUtils::LoadTexture(str.C_Str(), this->directory);
            texture.type = typeName;
            texture.path = str.C_Str();
            mTextures.append(texture);
            textures_loaded.append(texture); // store it as texture loaded for entire model, to ensure we won't unnecesery load duplicate mTextures.
        }
    }
    return mTextures;
}
