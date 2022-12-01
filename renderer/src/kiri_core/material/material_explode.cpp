/*
 * @Author: Xu.Wang 
 * @Date: 2020-03-17 17:48:50 
 * @Last Modified by:   Xu.Wang 
 * @Last Modified time: 2020-03-17 17:48:50 
 */
#include <kiri_core/material/material_explode.h>
#include <GLFW/glfw3.h>
void KiriMaterialExplode::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();
    mShader->Use();
}

void KiriMaterialExplode::Update()
{
    mShader->Use();
    mShader->SetFloat("time", (float)glfwGetTime());
}

KiriMaterialExplode::KiriMaterialExplode()
{
    mName = "explode";
    KiriMaterial::GeoShaderEnable();
    Setup();
}