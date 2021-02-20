/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:48:53 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:48:53 
 */
#include <kiri_core/material/material_g_buffer.h>

void KiriMaterialGBuffer::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
    mShader->Use();
}

void KiriMaterialGBuffer::Update()
{
    mShader->Use();
    mShader->SetBool("use_normal", use_normal && have_normal);
    mShader->SetBool("invert", !outside);
}

void KiriMaterialGBuffer::SetUseNormalMap(bool _use_normal)
{
    use_normal = _use_normal;
}

void KiriMaterialGBuffer::SetHaveNormalMap(bool _have_normal)
{
    have_normal = _have_normal;
}

void KiriMaterialGBuffer::SetOutside(bool _inside)
{
    outside = _inside;
}

KiriMaterialGBuffer::KiriMaterialGBuffer()
{
    mName = "g_buffer";
    use_normal = false;
    have_normal = false;
    outside = true;
    Setup();
}