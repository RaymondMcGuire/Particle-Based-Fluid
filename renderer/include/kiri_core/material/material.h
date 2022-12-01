/*** 
 * @Author: Xu.WANG
 * @Date: 2020-12-08 18:40:40
 * @LastEditTime: 2021-02-20 18:44:47
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriCore\include\kiri_core\material\material.h
 */

#ifndef _KIRI_MATERIAL_H_
#define _KIRI_MATERIAL_H_

#pragma once

#include <kiri_core/kiri_shader.h>

struct DEFAULT_DIRECTIONAL_LIGHT
{
    Vector3F direction = Vector3F(1.0f, 0.5f, 0.5f);
    Vector3F ambient = Vector3F(0.05f);
    Vector3F diffuse = Vector3F(0.6f);
    Vector3F specular = Vector3F(0.6f);
};

class KiriMaterial
{
public:
    virtual void Setup();
    virtual void Update() = 0;

    KiriShader *GetShader();
    String GetShaderName();

    virtual ~KiriMaterial()
    {
        delete mShader;
        mShader = nullptr;
    }

protected:
    String mName;
    KiriShader *mShader;

    void GeoShaderEnable()
    {
        mGeometryShaderEnbale = true;
    }

    void GeoShaderDisnable()
    {
        mGeometryShaderEnbale = false;
    }

    void BindGlobalUniformBufferObjects();

    DEFAULT_DIRECTIONAL_LIGHT mDefaultDirectLight;

private:
    bool mGeometryShaderEnbale = false;
};
typedef SharedPtr<KiriMaterial> KiriMaterialPtr;
#endif