/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-19 18:43:03
 * @LastEditTime: 2020-10-20 19:20:38
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\src\kiri_core\model\model_load_sampling.cpp
 */

#include <kiri_core/model/model_load_sampling.h>

void KiriModelLoadSampling::Load(String const &path)
{
    Assimp::Importer importer;

    const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        //cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << endl;
        return;
    }

    // get directory path
    directory = path.substr(0, path.find_last_of('/'));
    ProcessNode(scene->mRootNode, scene);
}

void KiriModelLoadSampling::ProcessNode(aiNode *node, const aiScene *scene)
{
    // process each mesh located at the current node
    for (UInt i = 0; i < node->mNumMeshes; i++)
    {
        // the node object only contains indices to index the actual objects in the scene.
        // the scene contains all the data, node is just to keep stuff organized (like relations between nodes).
        aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
        _meshes.append(ProcessMesh(mesh, scene));
    }
    // after we've processed all of the meshes (if any) we then recursively process each of the children nodes
    for (UInt i = 0; i < node->mNumChildren; i++)
    {
        ProcessNode(node->mChildren[i], scene);
    }
}

KiriMeshTriangle *KiriModelLoadSampling::ProcessMesh(aiMesh *mesh, const aiScene *scene)
{
    Array1Vec3F _vertices;
    Array1Vec3F _normals;
    Array1Vec3F _triangles;

    // vertex
    for (UInt i = 0; i < mesh->mNumVertices; i++)
    {
        _vertices.append(Vector3F(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z));
        //_normals.append(Vector3F(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z));
    }

    // face
    for (UInt i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];

        // TODO
        // face.mNumIndices === 3
        _triangles.append(Vector3F(face.mIndices[0], face.mIndices[1], face.mIndices[2]));
    }

    return new KiriMeshTriangle(_vertices, _normals, _triangles);
}