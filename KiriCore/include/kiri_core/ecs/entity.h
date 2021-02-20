/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 19:44:57
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\ecs\entity.h
 */

#ifndef _KIRI_ENTITY_H_
#define _KIRI_ENTITY_H_
#pragma once

#include <kiri_core/model/model.h>

class KiriEntity
{
public:
    KiriEntity() {}

    KiriEntity(size_t _id, KiriModelPtr _model, bool _have_normal = true, bool _outside = true, bool _static = true)
        : id(_id), model(_model), staticEntity(_static), outside(_outside), normalMap(_have_normal) {}

    KiriEntity(size_t _id, KiriModelPtr _model, KiriMaterialPtr _material, bool _have_normal = true, bool _outside = true, bool _static = true)
        : id(_id), material(_material), model(_model), staticEntity(_static), outside(_outside), normalMap(_have_normal) {}

    KiriModelPtr getModel() { return model; }
    KiriMaterialPtr GetMaterial() { return material; }
    Array1<Matrix4x4F> getModelMatrixs() { return modelMatrixs; }
    bool getOutside() { return outside; }
    bool getNormalMap() { return normalMap; }

private:
    size_t id;
    Array1<Matrix4x4F> modelMatrixs;

    KiriModelPtr model;
    KiriMaterialPtr material;

    bool staticEntity;
    bool outside;
    bool normalMap;
};

typedef SharedPtr<KiriEntity> KiriEntityPtr;
#endif