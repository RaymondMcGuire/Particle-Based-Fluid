/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:48:31 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-03-25 20:52:09
 */
#include <kiri_core/material/material_blinn_standard.h>

void KiriMaterialBlinnStandard::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();

    updateSetting();
}

//---------------PRIVATE METHOD----------------------------------------------
void KiriMaterialBlinnStandard::SetConstMaterial(STRUCT_MATERIAL _m)
{
    mShader->SetVec3("material.ambient", _m.ambient);
    mShader->SetVec3("material.diffuse", _m.diffuse);
    mShader->SetVec3("material.specular", _m.specular);
    mShader->SetFloat("material.shininess", _m.shininess);
}

void KiriMaterialBlinnStandard::updateSetting()
{
    mShader->Use();

    // texture map
    mShader->SetBool("textureMap.enable", bTextureMap);
    mShader->SetBool("textureMap.diffuse", bDiffuseTexure);
    mShader->SetBool("textureMap.specular", bSpecularTexure);
    mShader->SetBool("textureMap.normal", bNormalTexure);

    if (bTextureMap)
    {
        mShader->SetInt("materialTex.diffuse", 0);
        mShader->SetInt("materialTex.specular", 1);
        mShader->SetInt("materialTex.normal", 2);
        mShader->SetFloat("materialTex.shininess", 32.0f);
    }
    else
    {
        switch (constMaterial)
        {
        case T_EMERALD:
            SetConstMaterial(M_EMERALD);
            break;
        case T_JADE:
            SetConstMaterial(M_JADE);
            break;
        case T_OBSIDIAN:
            SetConstMaterial(M_OBSIDIAN);
            break;
        case T_PEARL:
            SetConstMaterial(M_PEARL);
            break;
        case T_RUBY:
            SetConstMaterial(M_RUBY);
            break;
        case T_TURQUOISE:
            SetConstMaterial(M_TURQUOISE);
            break;
        case T_BRASS:
            SetConstMaterial(M_BRASS);
            break;
        case T_BRONZE:
            SetConstMaterial(M_BRONZE);
            break;
        case T_CHROME:
            SetConstMaterial(M_CHROME);
            break;
        case T_COPPER:
            SetConstMaterial(M_COPPER);
            break;
        case T_GOLD:
            SetConstMaterial(M_GOLD);
            break;
        case T_SILVER:
            SetConstMaterial(M_SILVER);
            break;
        case T_BLACK_PLASTIC:
            SetConstMaterial(M_BLACK_PLASTIC);
            break;
        case T_CYAN_PLASTIC:
            SetConstMaterial(M_CYAN_PLASTIC);
            break;
        case T_GREEN_PLASTIC:
            SetConstMaterial(M_GREEN_PLASTIC);
            break;
        case T_RED_PLASTIC:
            SetConstMaterial(M_RED_PLASTIC);
            break;
        case T_WHITE_PLASTIC:
            SetConstMaterial(M_WHITE_PLASTIC);
            break;
        case T_YELLOW_PLASTIC:
            SetConstMaterial(M_YELLOW_PLASTIC);
            break;
        case T_BLACK_RUBBER:
            SetConstMaterial(M_BLACK_RUBBER);
            break;
        case T_CYAN_RUBBER:
            SetConstMaterial(M_CYAN_RUBBER);
            break;
        case T_GREEN_RUBBER:
            SetConstMaterial(M_GREEN_RUBBER);
            break;
        case T_RED_RUBBER:
            SetConstMaterial(M_RED_RUBBER);
            break;
        case T_WHITE_RUBBER:
            SetConstMaterial(M_WHITE_RUBBER);
            break;
        case T_YELLOW_RUBBER:
            SetConstMaterial(M_YELLOW_RUBBER);
            break;
        default:
            SetConstMaterial(M_GOLD);
            break;
        }
    }

    mShader->SetBool("inverse_normals", inverseNormal);
}

//---------------PRIVATE METHOD----------------------------------------------

void KiriMaterialBlinnStandard::Update()
{
    mShader->Use();

    //point light

    for (size_t i = 0; i < pointLights.size(); i++)
    {
        mShader->SetVec3("pointLights[" + std::to_string(i) + "].position", pointLights[i]->position);
        mShader->SetVec3("pointLights[" + std::to_string(i) + "].ambient", 1.0f, 1.0f, 1.0f);
        mShader->SetVec3("pointLights[" + std::to_string(i) + "].diffuse", pointLights[i]->diffuse);
        mShader->SetVec3("pointLights[" + std::to_string(i) + "].specular", 1.0f, 1.0f, 1.0f);
        mShader->SetFloat("pointLights[" + std::to_string(i) + "].constant", atten_constant);
        mShader->SetFloat("pointLights[" + std::to_string(i) + "].linear", atten_linear);
        mShader->SetFloat("pointLights[" + std::to_string(i) + "].quadratic", atten_quadratic);
    }

    mShader->SetInt("pointLightNum", (Int)pointLights.size());
    mShader->SetBool("gamma", gamma);
    mShader->SetBool("inverse_normals", inverseNormal);
    mShader->SetBool("textureMap.normal", bNormalTexure);
}

void KiriMaterialBlinnStandard::SetPointLights(Array1<KiriPointLightPtr> _pointLights)
{
    pointLights = _pointLights;
}

void KiriMaterialBlinnStandard::SetInverseNormal(bool _inverseNormal)
{
    inverseNormal = _inverseNormal;
    updateSetting();
}

void KiriMaterialBlinnStandard::SetNormalMap(bool _bNormalTexure)
{
    bNormalTexure = _bNormalTexure;
    updateSetting();
}

void KiriMaterialBlinnStandard::SetDiffuseTex(bool _bDiffuseTexure)
{
    bDiffuseTexure = _bDiffuseTexure;
    updateSetting();
}

void KiriMaterialBlinnStandard::SetSpecularTex(bool _bSpecularTexure)
{
    bSpecularTexure = _bSpecularTexure;
    updateSetting();
}

void KiriMaterialBlinnStandard::SetTextureMap(bool _bTextureMap)
{
    bTextureMap = _bTextureMap;
    updateSetting();
}

void KiriMaterialBlinnStandard::SetConsMaterial(KIRI_MATERIAL_CONSTANT_TYPE _type)
{
    constMaterial = _type;
    updateSetting();
}

void KiriMaterialBlinnStandard::SetAttenParams(float _constant, float _linear, float _quadratic)
{
    atten_constant = _constant;
    atten_linear = _linear;
    atten_quadratic = _quadratic;
}

KiriMaterialBlinnStandard::KiriMaterialBlinnStandard(bool _bTextureMap, bool _bDiffuseTexure, bool _bSpecularTexure, bool _bNormalTexure, bool _inverseNormal)
{
    mName = "blinn_standard";

    // default settings
    gamma = true;

    constMaterial = T_GOLD;
    atten_constant = 1.0f;
    atten_linear = 0.09f;
    atten_quadratic = 0.032f;

    bTextureMap = _bTextureMap;
    bDiffuseTexure = _bDiffuseTexure;
    bSpecularTexure = _bSpecularTexure;
    bNormalTexure = _bNormalTexure;

    inverseNormal = _inverseNormal;
    Setup();
}
