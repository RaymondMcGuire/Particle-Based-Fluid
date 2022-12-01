/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-19 11:25:28
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriCore\include\kiri_core\model\model_load_sampling.h
 */

#ifndef _KIRI_MODEL_LOAD_SAMPLING_H_
#define _KIRI_MODEL_LOAD_SAMPLING_H_

#pragma once

#include <kiri_define.h>
//3rd-party library
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <root_directory.h>

#include <kiri_core/model/model_load.h>
#include <kiri_core/mesh/mesh_triangle.h>

class KiriModelLoadSampling
{
public:
    String directory;

    KiriModelLoadSampling(String _name, String _folder = "models", String _ext = ".fbx")
        : folder(_folder), ext(_ext), mName(_name)
    {
		String path = String(DB_PBR_PATH) + folder + "/" + mName + "/" + mName + ext;

        if (RELEASE && PUBLISH)
        {
            //path = String(DB_PBR_PATH) + folder + "/" + mName + "/" + mName + ext;
            path = "./resources/" + folder + "/" + mName + "/" + mName + ext;
        }
        KIRI_LOG_INFO("Model Path={0}", path);
        Load(path);
    }

    Array1<KiriMeshTriangle *> meshes() const { return _meshes; }

private:
    String mName;
    String ext;
    String folder;
    bool gammaCorrection;

    void Load(String const &);
    void ProcessNode(aiNode *, const aiScene *);
    KiriMeshTriangle *ProcessMesh(aiMesh *, const aiScene *);

    Array1<KiriMeshTriangle *> _meshes;
};
typedef SharedPtr<KiriModelLoadSampling> KiriModelLoadSamplingPtr;
#endif