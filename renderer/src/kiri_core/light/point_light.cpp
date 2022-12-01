/*
 * @Author: Xu.Wang
 * @Date: 2020-03-17 18:07:03
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-15 01:49:15
 */
#include <kiri_core/light/point_light.h>

void KiriPointLight::SetModel()
{
    material->SetColor(diffuse);
    model->SetMaterial(material);
    model->BindShader();
    model->ResetModelMatrix();

    model->Translate(Vector3F(position.x, position.y, position.z));
    model->Scale(Vector3F(0.1f));
}

KiriPointLight::KiriPointLight()
{
    mName = "point_light";
    position = Vector3F();
    diffuse = Vector3F();
    material = std::make_shared<KiriMaterialLamp>(diffuse);
    model = std::make_shared<KiriCube>();
}

KiriPointLight::KiriPointLight(Vector3F _pos, Vector3F _diffuse)
{
    mName = "point_light";
    position = _pos;
    diffuse = _diffuse;
    material = std::make_shared<KiriMaterialLamp>(diffuse);
    model = std::make_shared<KiriCube>();
}

void KiriPointLight::Draw()
{
    SetModel();
    KiriLight::Draw();
}