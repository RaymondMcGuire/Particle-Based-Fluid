/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 19:36:50
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\model\model_load_obj.h
 */

#ifndef _KIRI_MODEL_LOAD_OBJ_H_
#define _KIRI_MODEL_LOAD_OBJ_H_
#pragma once
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <kiri_core/model/model_load.h>
#include <kiri_utils.h>

#include <kiri_core/mesh/mesh.h>

class KiriModelLoadObj : public KiriModelLoad
{
public:
    String directory;

    KiriModelLoadObj(String const &path, bool _instancing = false, Array1<Matrix4x4F> _trans4 = {})
        : trans4(_trans4)
    {
        instancing = _instancing;
        Load(path);
    }

    void Draw() override;

private:
    // load model from file
    void Load(String const &);
    void ProcessNode(aiNode *, const aiScene *);
    KiriMesh *ProcessMesh(aiMesh *, const aiScene *);
    Array1<Texture> LoadMaterialTextures(aiMaterial *, aiTextureType, String);

    Array1<Texture> textures_loaded;
    Array1<KiriMesh *> meshes;

    Array1<Matrix4x4F> trans4;
};
typedef SharedPtr<KiriModelLoadObj> KiriModelLoadObjPtr;
#endif