/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:59:36
 * @FilePath: \Kiri\renderer\include\kiri_core\ecs\entity.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
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
        : id(_id), mModel(_model), mStaticEntity(_static), mOutside(_outside), mNormalMap(_have_normal)
    {

        AddModelMatrix(_model->GetModelMatrix());
    }

    KiriEntity(size_t _id, KiriModelPtr _model, KiriMaterialPtr _material, bool _have_normal = true, bool _outside = true, bool _static = true)
        : id(_id), mMat(_material), mModel(_model), mStaticEntity(_static), mOutside(_outside), mNormalMap(_have_normal)
    {

        AddModelMatrix(_model->GetModelMatrix());
    }

    void AddModelMatrix(Matrix4x4F _modelmatrix)
    {
        mModelMatrixs.append(_modelmatrix);
    }

    void clearModelMatrix()
    {
        mModelMatrixs.clear();
    }

    KiriModelPtr GetModel() { return mModel; }
    KiriMaterialPtr GetMaterial() { return mMat; }
    Array1<Matrix4x4F> GetModelMatrixs() { return mModelMatrixs; }
    bool GetOutside() { return mOutside; }
    bool GetNormalMap() { return mNormalMap; }

    void SetModelMatrix(size_t _idx, Matrix4x4F _modelmatrix)
    {
        mModelMatrixs[_idx] = _modelmatrix;
    }

private:
    size_t id;
    Array1<Matrix4x4F> mModelMatrixs;

    KiriModelPtr mModel;
    KiriMaterialPtr mMat;

    bool mStaticEntity;
    bool mOutside;
    bool mNormalMap;
};

typedef SharedPtr<KiriEntity> KiriEntityPtr;
#endif