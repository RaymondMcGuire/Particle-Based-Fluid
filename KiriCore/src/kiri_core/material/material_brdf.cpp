/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:48:34 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:48:34 
 */
#include <kiri_core/material/material_brdf.h>

void KiriMaterialBRDF::Setup()
{
    KiriMaterial::Setup();
    mShader->Use();
}

void KiriMaterialBRDF::Update()
{
    mShader->Use();
}

KiriMaterialBRDF::KiriMaterialBRDF()
{
    mName = "brdf";
    Setup();
}