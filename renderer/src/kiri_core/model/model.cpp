/***
 * @Author: Xu.WANG
 * @Date: 2021-02-22 11:23:43
 * @LastEditTime: 2021-04-07 14:18:51
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriCore\src\kiri_core\model\model.cpp
 */

#include <kiri_core/model/model.h>

KiriMaterialPtr KiriModel::GetMaterial() { return mMat; }

void KiriModel::SetMaterial(KiriMaterialPtr _material)
{
    mMat = _material;
}

void KiriModel::BindShader()
{
    mMat->Update();
}

void KiriModel::SetModelMatrix(Matrix4x4F _modelMatrix)
{
    mModelMatrix = _modelMatrix;
}

Matrix4x4F KiriModel::GetModelMatrix()
{
    return mModelMatrix;
}

void KiriModel::ResetModelMatrix()
{
    mModelMatrix = Matrix4x4F::identity();
}

void KiriModel::Draw()
{
    mMat->GetShader()->SetMat4("model", mModelMatrix);
}

void KiriModel::Translate(Vector3F _v)
{
    mModelMatrix = Matrix4x4F::translationMatrix(_v).transposed().mul(mModelMatrix);
}
void KiriModel::Scale(Vector3F _v)
{
    mModelMatrix = Matrix4x4F::scaleMatrix(_v).transposed().mul(mModelMatrix);
}

void KiriModel::Rotate(float radian, Vector3F axis)
{
    mModelMatrix = Matrix4x4F::rotationMatrix(axis, radian).transposed().mul(mModelMatrix);
}