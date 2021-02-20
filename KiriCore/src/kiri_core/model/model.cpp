/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 19:27:58 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-24 21:36:21
 */
#include <kiri_core/model/model.h>
#include <kiri_utils.h>

KiriMaterialPtr KiriModel::GetMaterial() { return material; }

void KiriModel::SetMaterial(KiriMaterialPtr _material)
{
    material = _material;
}

void KiriModel::BindShader()
{
    material->Update();
}

void KiriModel::SetModelMatrix(Matrix4x4F _modelMatrix)
{
    modelMatrix = _modelMatrix;
}

Matrix4x4F KiriModel::GetModelMatrix()
{
    return modelMatrix;
}

void KiriModel::ResetModelMatrix()
{
    modelMatrix = Matrix4x4F::identity();
}

void KiriModel::Draw()
{
    material->GetShader()->SetMat4("model", modelMatrix);
}

void KiriModel::Translate(Vector3F _v)
{
    modelMatrix = Matrix4x4F::translationMatrix(_v).transposed().mul(modelMatrix);
}
void KiriModel::Scale(Vector3F _v)
{
    modelMatrix = Matrix4x4F::scaleMatrix(_v).transposed().mul(modelMatrix);
}

void KiriModel::Rotate(float radian, Vector3F axis)
{
    modelMatrix = Matrix4x4F::rotationMatrix(axis, radian).transposed().mul(modelMatrix);
}