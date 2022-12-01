/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:28:23
 * @FilePath: \core\include\kiri_core\model\model_load_obj.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _KIRI_MODEL_LOAD_OBJ_H_
#define _KIRI_MODEL_LOAD_OBJ_H_
#pragma once
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <kiri_core/model/model_load.h>

#include <kiri_core/mesh/mesh.h>

class KiriModelLoadObj : public KiriModelLoad
{
public:
    String directory;

    KiriModelLoadObj(String const &path, bool _instancing = false, Array1<Matrix4x4F> _trans4 = {})
        : trans4(_trans4)
    {
        mInstance = _instancing;
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