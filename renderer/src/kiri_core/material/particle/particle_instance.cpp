/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-20 02:04:30 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-20 02:13:10
 */
#include <kiri_core/material/particle/particle_instance.h>

void KiriMaterialParticleInstance::Setup()
{
    KiriMaterial::Setup();
    BindGlobalUniformBufferObjects();

    if (!_singleColor)
    {
        glActiveTexture(GL_TEXTURE0);
        glGenTextures(1, &color_tbo);
        glBindTexture(GL_TEXTURE_BUFFER, color_tbo);

        glGenBuffers(1, &color_buffer);
        glBindBuffer(GL_TEXTURE_BUFFER, color_buffer);
        glBufferData(GL_TEXTURE_BUFFER, sizeof(float) * 4 * _colorArray.size(), NULL, GL_DYNAMIC_DRAW);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, color_buffer);
        glBindBuffer(GL_TEXTURE_BUFFER, 0);
    }
}

void KiriMaterialParticleInstance::Update()
{
    mShader->Use();
    mShader->SetInt("color_tbo", 0);
    mShader->SetBool("singleColor", _singleColor);
    mShader->SetVec3("dirLightPos", _dirLightPos);
    mShader->SetVec3("lightColor", _lightColor);

    if (_singleColor)
    {
        mShader->SetVec3("particleColor", _particleColor);
    }
    else
    {
        // Vector4F color = _colorArray[0];
        // KIRI_INFO << "Phase Color=" << color.x << "," << color.y << "," << color.z;
        glBindBuffer(GL_TEXTURE_BUFFER, color_buffer);
        glBufferData(GL_TEXTURE_BUFFER, sizeof(float) * 4 * _colorArray.size(), _colorArray.data(), GL_DYNAMIC_DRAW);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, color_tbo);
    }
}

KiriMaterialParticleInstance::KiriMaterialParticleInstance()
{
    mName = "particle_instance";
    _singleColor = true;
    Setup();
}

KiriMaterialParticleInstance::KiriMaterialParticleInstance(bool singleColor, Array1Vec4F colorArray)
{
    mName = "particle_instance";
    _singleColor = singleColor;
    _colorArray = colorArray;
    Setup();
}

void KiriMaterialParticleInstance::updateColorArray(Array1Vec4F colorArray)
{
    _colorArray = colorArray;
}

void KiriMaterialParticleInstance::SetDirLightPos(Vector3F dlp)
{
    _dirLightPos = dlp;
}
void KiriMaterialParticleInstance::SetParticleColor(Vector3F pc)
{
    _particleColor = pc;
}
void KiriMaterialParticleInstance::SetLightColor(Vector3F lc)
{
    _lightColor = lc;
}
