/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-18 01:04:15
 * @LastEditTime: 2021-02-20 19:39:06
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\model\model.h
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
    KiriMaterialPtr material = NULL;
    Matrix4x4F modelMatrix = Matrix4x4F::identity();

    bool instancing = false;
    bool static_obj = true;
};
typedef SharedPtr<KiriModel> KiriModelPtr;
#endif