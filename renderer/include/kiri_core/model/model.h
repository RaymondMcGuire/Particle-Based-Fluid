/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-22 11:06:26
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-10-22 14:30:18
 * @FilePath: \core\include\kiri_core\model\model.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _KIRI_MODEL_H_
#define _KIRI_MODEL_H_
#pragma once
#include <kiri_core/material/material.h>

class KiriModel
{
public:
    virtual void Draw();

    void SetMaterial(KiriMaterialPtr);
    KiriMaterialPtr GetMaterial();
    void BindShader();

    void Translate(Vector3F);
    void Scale(Vector3F);
    void Rotate(float radian, Vector3F axis);
    void SetModelMatrix(Matrix4x4F);
    Matrix4x4F GetModelMatrix();
    void ResetModelMatrix();

protected:
    KiriMaterialPtr mMat = NULL;
    Matrix4x4F mModelMatrix = Matrix4x4F::identity();

    bool mInstance = false;
    bool mStaticObj = true;
};
typedef SharedPtr<KiriModel> KiriModelPtr;
#endif